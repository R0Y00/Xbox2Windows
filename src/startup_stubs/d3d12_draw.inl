// Experimental (docs/startup-runtime.md): draws through translated pipelines.
// Included inside d3d12_xs_device.cpp's anonymous namespace, after the Pass helpers.
//
// A machine-code pipeline is looked up by the hash of its two code blobs and its root signature in
// X2W_SHADER_CACHE: <key>.txt (manifest), <key>.vs.cso and <key>.ps.cso, produced offline by
// out/gamerun/gfx2hlsl.py. A miss writes req_<key>.{a,b,root} there for the next offline pass.
// The translated shaders use their own root signature: [0] 32-bit constants ("entry" values such
// as the base vertex), an optional SRV table and an optional sampler table. Each draw copies what the title bound (vertex streams,
// buffers named by the hardware descriptors in its tables, textures behind view tokens) into an
// upload ring / a private descriptor heap.

constexpr int draw_srv_limit = 64;

struct Translation {
    struct Srv { int kind, source, a, b; };  // kind 0 buffer, 1 2D texture, 2 2D array, 3 3D texture, 4 1D texture; source 0 table(param a, byte b), 1 vertex stream a, 2 root parameter a
    enum UdKind { rect, base_vertex, start_instance, root_constant, stride, mrt_bits, buf_word3, uav_word3, uav_base, uav_stride, d16_unorm16, desc_word3, desc_word };
    struct Ud { int kind, a, b; };
    struct Uav { int kind, source, a, b; };
    int state = 0;                           // 1 ready, 2 unavailable
    ULONGLONG retry_manifest_at = 0;         // Only an absent manifest can become ready on a later draw.
    std::vector<Srv> srvs;
    std::vector<Ud> uds;
    std::vector<Uav> uavs;                   // u0.. in their own table (root parameter uav_parameter)
    unsigned uav_base_wanted = 0;            // bit per uav that has a `udsrc uavbase` value: a misaligned view may start below it
    UINT uav_parameter = 0;
    bool heap = false;                       // `heap 1`: heapmem, vinfo and the view-id region (root parameters heap_parameter..+2)
    UINT heap_parameter = 0;
    bool heap_views_logged = false;
    bool heapwords = false;
    UINT vdesc_parameter = 0;
    x2w_startup::TranslatedSamplers samplers;
    x2w_startup::ExportWriteMasks write_masks;  // manifest 'wmask': channels each export leaves unwritten (#1674)
    x2w_startup::D16ImageSrvs d16_srvs;         // manifest 'd16srv': texture views that must match proven D16 formats (#2023)
    ID3DBlob* vs = nullptr;
    ID3DBlob* ps = nullptr;
    ID3DBlob* ms = nullptr;
    ID3D12RootSignature* root = nullptr;
    std::optional<bool> independent_indirect_entry;
    std::optional<bool> indexed_indirect_entry;
    bool indexed_rect_geometry_attempted = false;
    x2w_startup::ComPtr<ID3DBlob> indexed_rect_geometry;
    int indirect_base_word = -1;
    x2w_startup::ComPtr<ID3D12CommandSignature> indexed_signature;
    std::unordered_map<u64, ID3D12PipelineState*> pipelines;  // by target formats
    u64 key = 0;
    LONG logged_draws = 0, logged_bindings = 0, logged_large = 0;  // diagnostics budget per translation
};

// Per title pipeline: its translation and the blend state found in its state block (if any).
struct BoundPipeline {
    Translation* translation = nullptr;
    const std::vector<RootParameter>* guest_root = nullptr;
    bool has_blend = false;
    D3D12_BLEND_DESC blend{};
    bool has_depth_stencil = false;
    D3D12_DEPTH_STENCIL_DESC depth_stencil{};
    bool depth_from_registers = false;  // X2W_DEPTH_STATE: depth_stencil was decoded from state block A (slot 57)
    bool has_stencil_ref = false;       // ... and its stencil test is on: stencil_ref goes on the list with each draw
    std::uint8_t stencil_ref = 0;
    bool has_rasterizer = false;
    D3D12_RASTERIZER_DESC rasterizer{};
    DXGI_FORMAT dsv_format = DXGI_FORMAT_UNKNOWN;
    std::unordered_map<u64, ID3D12PipelineState*> pipelines;  // by target formats
    bool has_words = false;          // state block A read (ds / rs / masked_blend valid where .ok)
    bool mesh_layout = false;        // A+0x004 == 6: the fields sit 0x54 later (read_words handled it)
    bool blend_overridden = false;   // a slot 55 blend patch or a stream blend subobject replaces the register words
    bool derived_ds = false, derived_rs = false;  // a slot 55 depth-stencil / rasterizer part replaces A's
    x2w_pipeline_state::DepthStencil ds;
    x2w_pipeline_state::Raster rs;
    x2w_pipeline_state::Blend masked_blend;
    D3D12_DEPTH_STENCIL_DESC patch_ds{};
    D3D12_RASTERIZER_DESC patch_rs{};
    std::uint32_t raw_mode = 0, raw_clip = 0;
};

namespace draw_state {
enum Counter {
    decoded, decoded_mesh, class_depth_off, class_depth_read, class_depth_write, class_stencil, class_cull, class_blend, class_masked,
    class_dual, refused_depth, refused_raster, refused_blend, derived_ds, derived_rs, derived_invalid, depth_bounds_ignored,
    stencil_masks_differ, clip_split, bias_rounded, conservative_dropped, pso_new, pso_failed_fallback, pso_failed_both,
    draws_depth_no_target, draws_stencil_no_plane, draws_read_only, draws_rect_cull_off, dual_multi_target, draws_depth_uncleared, count
};
LONG n[count];
void add(Counter c) { InterlockedIncrement(&n[c]); }
const x2w_pipeline_state::Switches& switches()
{
    static const x2w_pipeline_state::Switches s = [] {
        const auto v = x2w_pipeline_state::switches_from_env();
        log("x2w-draw-state switches: depth %d cull %d (flip %d) blend masks %d depth clears %d depth clip %d (X2W_DRAW_STATE_DEPTH / "
            "_CULL / _CULL_FLIP / _BLEND / _DEPTH_CLEAR / _DEPTHCLIP, X2W_DRAW_STATE=0 all off)", v.depth, v.cull, v.flip, v.blend,
            v.depth_clear, v.depthclip);
        return v;
    }();
    return s;
}
}  // namespace draw_state

// One line per 300 frames (counters reset here).
void report_draw_state()
{
    using namespace draw_state;
    LONG v[count];
    for (int i = 0; i < count; ++i) v[i] = InterlockedExchange(&n[i], 0);
    const auto& s = switches();
    log("x2w-draw-state 300 frames (depth %d cull %d flip %d masks %d): pipelines decoded %ld (mesh layout %ld) [depth off %ld, read %ld, "
        "write %ld, stencil %ld, cull %ld, blend %ld, masked %ld, dual-source %ld]; refused depth %ld raster %ld blend %ld; derived parts "
        "depth-stencil %ld rasterizer %ld invalid %ld; flagged: depth bounds ignored %ld, stencil back masks differ %ld, one zclip "
        "%ld, bias rounded %ld, conservative dropped %ld; PSOs new-state %ld, failed->old state %ld, failed both %ld; draws: depth/"
        "stencil test without depth target %ld, stencil without plane %ld, read-only target writes off %ld, rect list cull off %ld, "
        "dual-source with >1 target %ld, depth/stencil test on a never-cleared depth target off %ld",
        s.depth, s.cull, s.flip, s.blend, v[decoded], v[decoded_mesh], v[class_depth_off], v[class_depth_read], v[class_depth_write],
        v[class_stencil], v[class_cull], v[class_blend], v[class_masked], v[class_dual], v[refused_depth], v[refused_raster],
        v[refused_blend], v[derived_ds], v[derived_rs], v[derived_invalid], v[depth_bounds_ignored], v[stencil_masks_differ],
        v[clip_split], v[bias_rounded], v[conservative_dropped], v[pso_new], v[pso_failed_fallback], v[pso_failed_both],
        v[draws_depth_no_target], v[draws_stencil_no_plane], v[draws_read_only], v[draws_rect_cull_off], v[dual_multi_target],
        v[draws_depth_uncleared]);
    if (s.depth_clear)
        log("x2w-draw-state 300 frames: depth clears (slot 51) forwarded %ld; dropped: list not open %ld, not a depth view %ld, depth "
            "outside [0, 1] %ld, nothing to clear %ld", InterlockedExchange(&depth_clears[0], 0), InterlockedExchange(&depth_clears[1], 0),
            InterlockedExchange(&depth_clears[2], 0), InterlockedExchange(&depth_clears[3], 0), InterlockedExchange(&depth_clears[4], 0));
}
int load_mesh_stages(Translation& tr, const std::string& manifest);
bool mesh_issue_active(Wrapper* self);
ID3D12PipelineState* mesh_pso_for(const Translation& tr);

// Looks for a desktop-layout D3D12_BLEND_DESC inside a state block: two flags, then eight render
// target entries whose fields are all inside their enumerations.
bool find_blend(const std::vector<unsigned char>& block, D3D12_BLEND_DESC* out, std::size_t* where)
{
    for (std::size_t at = 0; at + sizeof(D3D12_BLEND_DESC) <= block.size(); at += 4) {
        D3D12_BLEND_DESC d;
        std::memcpy(&d, block.data() + at, sizeof(d));
        bool ok = (d.AlphaToCoverageEnable == 0 || d.AlphaToCoverageEnable == 1) && (d.IndependentBlendEnable == 0 || d.IndependentBlendEnable == 1);
        for (int i = 0; ok && i < 8; ++i) {
            const auto& t = d.RenderTarget[i];
            ok = (t.BlendEnable == 0 || t.BlendEnable == 1) && (t.LogicOpEnable == 0 || t.LogicOpEnable == 1) && t.SrcBlend >= 1 &&
                 t.SrcBlend <= 19 && t.DestBlend >= 1 && t.DestBlend <= 19 && t.BlendOp >= 1 && t.BlendOp <= 5 && t.SrcBlendAlpha >= 1 &&
                 t.SrcBlendAlpha <= 19 && t.DestBlendAlpha >= 1 && t.DestBlendAlpha <= 19 && t.BlendOpAlpha >= 1 && t.BlendOpAlpha <= 5 &&
                 static_cast<unsigned>(t.LogicOp) <= 15 && t.RenderTargetWriteMask <= 15;
        }
        if (ok) {
            *out = d;
            *where = at;
            return true;
        }
    }
    return false;
}

SRWLOCK draw_lock = SRWLOCK_INIT;
void log_thread_stack(DWORD thread_id, const char* what)
{
    HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, thread_id);
    if (!thread) return;
    DWORD64 frames[40] = {};
    int depth = 0;
    if (SuspendThread(thread) != static_cast<DWORD>(-1)) {
        CONTEXT context{};
        context.ContextFlags = CONTEXT_FULL;
        if (GetThreadContext(thread, &context)) {
            for (; depth < 40 && context.Rip; ++depth) {
                frames[depth] = context.Rip;
                DWORD64 image = 0;
                PRUNTIME_FUNCTION function = RtlLookupFunctionEntry(context.Rip, &image, nullptr);
                if (!function) {
                    // A leaf function: the return address is at rsp.
                    if (IsBadReadPtr(reinterpret_cast<void*>(context.Rsp), 8)) break;
                    context.Rip = *reinterpret_cast<DWORD64*>(context.Rsp);
                    context.Rsp += 8;
                    continue;
                }
                void* handler_data = nullptr;
                DWORD64 establisher = 0;
                RtlVirtualUnwind(UNW_FLAG_NHANDLER, image, context.Rip, function, &context, &handler_data, &establisher, nullptr);
            }
        }
        ResumeThread(thread);
    }
    CloseHandle(thread);
    char text[2400];
    int at = 0;
    for (int i = 0; i < depth && at < static_cast<int>(sizeof(text)) - 80; ++i) {
        HMODULE module = nullptr;
        char name[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCSTR>(frames[i]), &module) && module) {
            GetModuleFileNameA(module, name, MAX_PATH);
            const char* slash = std::strrchr(name, '\\');
            at += std::snprintf(text + at, sizeof(text) - at, " %s+0x%llx", slash ? slash + 1 : name,
                                static_cast<unsigned long long>(frames[i] - reinterpret_cast<DWORD64>(module)));
        } else {
            at += std::snprintf(text + at, sizeof(text) - at, " 0x%llx", static_cast<unsigned long long>(frames[i]));
        }
    }
    log("x2w-draw %s thread %lu stack:%s", what, thread_id, text);
}
volatile LONG draw_lock_line, draw_lock_thread;
volatile LONG64 draw_lock_since;
void draw_lock_taken(int line)
{
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    draw_lock_line = line;
    draw_lock_thread = static_cast<LONG>(GetCurrentThreadId());
    draw_lock_since = now.QuadPart;
    static const HANDLE monitor = CreateThread(nullptr, 0, [](void*) -> DWORD {
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        LONG64 reported = 0;
        for (;;) {
            Sleep(5000);
            if (TryAcquireSRWLockExclusive(&draw_lock)) {
                ReleaseSRWLockExclusive(&draw_lock);
                continue;
            }
            const LONG64 since = draw_lock_since;
            LARGE_INTEGER now;
            QueryPerformanceCounter(&now);
            const double held = static_cast<double>(now.QuadPart - since) / static_cast<double>(frequency.QuadPart);
            // Two more looks 5 s apart show whether the holder is moving.
            static int looks;
            if (held >= 10.0 && since == reported && looks < 2) {
                ++looks;
                log_thread_stack(static_cast<DWORD>(draw_lock_thread), "draw_lock holder (again)");
            }
            if (held >= 10.0 && since != reported) {
                looks = 0;
                reported = since;
                log("x2w-draw draw_lock held %.1f s by thread %ld (taken at line %ld)", held, draw_lock_thread, draw_lock_line);
                log_thread_stack(static_cast<DWORD>(draw_lock_thread), "draw_lock holder");
            }
        }
    }, nullptr, 0, nullptr);
    (void)monitor;
}
namespace draw_lock_times {
struct Site {
    volatile LONG id;
    volatile LONG64 holds, hold_ticks, hold_max, waits, wait_ticks, wait_max;
};
constexpr int site_count = 128;
Site sites[site_count];
volatile LONG overflow;     // acquisitions whose site found no free slot
volatile LONG hold_open;    // written by the holder only: draw_lock_taken ran and the release has not been counted
bool on()
{
    static const bool value = x2w_local::env_on("X2W_DRAW_LOCK_TIMES");
    return value;
}
Site* site(LONG id)
{
    const unsigned start = static_cast<unsigned>(id) * 2654435761u;
    for (int i = 0; i < site_count; ++i) {
        Site& s = sites[(start + static_cast<unsigned>(i)) % site_count];
        LONG seen = s.id;
        if (seen == id) return &s;
        if (seen == 0) {
            seen = InterlockedCompareExchange(&s.id, id, 0);
            if (seen == 0 || seen == id) return &s;
        }
    }
    InterlockedIncrement(&overflow);
    return nullptr;
}
void raise_max(volatile LONG64& max, LONG64 value)
{
    for (LONG64 seen = max; value > seen;) {
        const LONG64 before = InterlockedCompareExchange64(&max, value, seen);
        if (before == seen) break;
        seen = before;
    }
}
void note_wait(LONG id, LONG64 ticks)
{
    if (Site* s = site(id)) {
        InterlockedIncrement64(&s->waits);
        InterlockedAdd64(&s->wait_ticks, ticks);
        raise_max(s->wait_max, ticks);
    }
}
const char* file_of(LONG id)
{
    switch (id / 100000) {
    case 0: return "xs_device";
    case 1: return "draw";
    case 2: return "compute";
    case 3: return "mesh";
    case 4: return "rawaddr";
    case 9: return id >= 999000 ? "fixture" : "memcut_release";
    default: return "?";
    }
}
// Every 300 frames (queue_present) and from the fixture export: sites by hold time, then reset.
void report(const char* when)
{
    if (!on()) return;
    static LONG64 frequency = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return f.QuadPart;
    }();
    struct Row { LONG id; LONG64 holds, hold_ticks, hold_max, waits, wait_ticks, wait_max; };
    Row rows[site_count];
    int count = 0;
    for (Site& s : sites) {
        if (!s.id) continue;
        rows[count++] = {s.id, InterlockedExchange64(&s.holds, 0), InterlockedExchange64(&s.hold_ticks, 0), InterlockedExchange64(&s.hold_max, 0),
                         InterlockedExchange64(&s.waits, 0), InterlockedExchange64(&s.wait_ticks, 0), InterlockedExchange64(&s.wait_max, 0)};
        if (!rows[count - 1].holds && !rows[count - 1].waits) --count;
    }
    std::sort(rows, rows + count, [](const Row& a, const Row& b) { return a.hold_ticks > b.hold_ticks; });
    const auto ms = [](LONG64 ticks) { return static_cast<double>(ticks) * 1000.0 / static_cast<double>(frequency); };
    LONG64 hold_total = 0, wait_total = 0;
    for (int i = 0; i < count; ++i) hold_total += rows[i].hold_ticks, wait_total += rows[i].wait_ticks;
    char line[3072];
    int n = std::snprintf(line, sizeof(line), "x2w-draw lock times (%s, X2W_DRAW_LOCK_TIMES): held %.1f ms, waited %.1f ms over %d sites%s; "
                          "site id file holds/total ms/max ms, waits/total ms/max ms:", when, ms(hold_total), ms(wait_total), count,
                          overflow ? " (site table full: some not counted)" : "");
    for (int i = 0; i < count && n < static_cast<int>(sizeof(line)) - 160; ++i) {
        const Row& r = rows[i];
        n += std::snprintf(line + n, sizeof(line) - n, " %ld %s %lld/%.1f/%.2f, %lld/%.1f/%.2f;", r.id, file_of(r.id), r.holds, ms(r.hold_ticks),
                           ms(r.hold_max), r.waits, ms(r.wait_ticks), ms(r.wait_max));
    }
    log("%s", line);
}
}  // namespace draw_lock_times
void lock_draw(int line)
{
    if (!TryAcquireSRWLockExclusive(&draw_lock)) {
        Timed waited(time_draw_lock_wait);
        if (draw_lock_times::on()) {
            LARGE_INTEGER before, after;
            QueryPerformanceCounter(&before);
            AcquireSRWLockExclusive(&draw_lock);
            QueryPerformanceCounter(&after);
            draw_lock_times::note_wait(line, after.QuadPart - before.QuadPart);
        } else {
            AcquireSRWLockExclusive(&draw_lock);
        }
    }
    draw_lock_taken(line);
    if (draw_lock_times::on()) draw_lock_times::hold_open = 1;
}
void release_exclusive_timed(PSRWLOCK lock)
{
    if (lock == &draw_lock && draw_lock_times::on() && draw_lock_times::hold_open &&
        draw_lock_thread == static_cast<LONG>(GetCurrentThreadId())) {
        draw_lock_times::hold_open = 0;
        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        const LONG64 ticks = now.QuadPart - draw_lock_since;
        if (draw_lock_times::Site* s = draw_lock_times::site(draw_lock_line)) {
            InterlockedIncrement64(&s->holds);
            InterlockedAdd64(&s->hold_ticks, ticks);
            draw_lock_times::raise_max(s->hold_max, ticks);
        }
    }
    ReleaseSRWLockExclusive(lock);
}
#define ReleaseSRWLockExclusive(lock) release_exclusive_timed(lock)
bool draw_lock_split()
{
    static const bool on = x2w_local::env_on("X2W_DRAW_LOCK_SPLIT");
    return on;
}
// Takes draw_lock for its scope when `split` (the lock is not held then); nothing otherwise (it is held).
struct SplitRelock {
    bool split;
    SplitRelock(bool s, int line) : split(s) { if (split) lock_draw(line); }
    ~SplitRelock() { if (split) ReleaseSRWLockExclusive(&draw_lock); }
    SplitRelock(const SplitRelock&) = delete;
    SplitRelock& operator=(const SplitRelock&) = delete;
};
// Fixture hook (startup_draw_lock_times): op 0 takes draw_lock at site 999000 + b, holds it a microseconds (spin)
// and releases it; op 1 writes the report now ("fixture").
extern "C" __declspec(dllexport) u64 WINAPI x2w_draw_lock_times_test(int op, u64 a, u64 b)
{
    if (op == 1) {
        draw_lock_times::report("fixture");
        return draw_lock_times::on() ? 1 : 0;
    }
    if (op != 0) return 0;
    lock_draw(999000 + static_cast<int>(b));
    LARGE_INTEGER start, now, frequency;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    do QueryPerformanceCounter(&now);
    while (static_cast<u64>(now.QuadPart - start.QuadPart) * 1000000ull < a * static_cast<u64>(frequency.QuadPart));
    ReleaseSRWLockExclusive(&draw_lock);
    return 1;
}
std::unordered_map<const Wrapper*, BoundPipeline*>& translation_by_pipeline() { static std::unordered_map<const Wrapper*, BoundPipeline*> m; return m; }
std::unordered_map<u64, Translation*>& translation_by_key() { static std::unordered_map<u64, Translation*> m; return m; }
LONG translated_draws, untranslated_draws, translation_ready, translation_missing, null_bindings;
LONG formatted_draws_refused;  // draws whose formatted buffer load had no resolved descriptor word 3 (also untranslated)
LONG null_by_cause[x2w_local::null_cause_count];  // null_bindings split by cause; the parts add up to it
LONG rt_loaded_word3, rt_loaded_uav, rt_loaded_heap;
LONG rt_word3_set, rt_word3_zero;
LONG rt_uav_draws, rt_uav_buf_bound, rt_uav_buf_unresolved, rt_uav_buf_misaligned, rt_srv_in_place, rt_srv_stale_copy;
LONG rt_heap_draws, rt_heap_skipped;
LONG rt_uav_tex[6];  // texture uavs: [0] bound; [1..5] draws skipped (unresolved, flag, shape, format, typed store)
LONG rt_written_evicted;  // GPU-written ranges dropped (registry full)
// GPU-ordered CopyBufferRegion (slot 15, X2W_D3D12_COPY_BUFFER=gpu): copies run on the GPU, and record-time
// fallbacks by reason (the record-time copy of d3d12_copy_buffer.inl).
enum CopyFallback { cf_no_list, cf_no_title_memory, cf_range, cf_unaligned, cf_overlap, cf_dest_unresolved, cf_source_unresolved,
                    cf_no_pipeline, cf_copy_list_same, cf_reasons };
const char* const copy_fallback_names[cf_reasons] = {"list not recording", "no title memory", "size or offset out of range",
    "offset or size not a multiple of 4", "overlapping ranges in one window", "destination not in one window",
    "source not in one window", "copy pipeline unavailable", "copy-type list, source and destination in one window"};
LONG cg_gpu, cg_same_window, cg_copy_list, cg_record_too, cg_fallback[cf_reasons];
LONG64 cg_bytes;
std::size_t gpu_written_count();  // draw_lock held
LONG rt_report_frame = -1;
void report_texture_census();
void report_draw_rt()  // draw_lock held
{
    report_texture_census();
    if (frame_number % 300 || frame_number == rt_report_frame) return;
    LONG fallbacks = 0;
    for (LONG f : cg_fallback) fallbacks += f;
    if (cg_gpu || fallbacks) {
        rt_report_frame = frame_number;
        log("x2w-copy-gpu totals: frame %ld; on the GPU %ld (same window %ld, copy-type lists %ld, %lld KiB; also copied at record time %ld); "
            "record-time fallback %ld (list %ld, title memory %ld, range %ld, unaligned %ld, overlap %ld, dest unresolved %ld, source unresolved %ld, "
            "pipeline %ld, copy list same window %ld); gpu-written ranges %zu evicted %ld",
            frame_number, cg_gpu, cg_same_window, cg_copy_list, cg_bytes >> 10, cg_record_too, fallbacks, cg_fallback[0], cg_fallback[1],
            cg_fallback[2], cg_fallback[3], cg_fallback[4], cg_fallback[5], cg_fallback[6], cg_fallback[7], cg_fallback[8], gpu_written_count(),
            rt_written_evicted);
    }
    if (!(rt_loaded_word3 | rt_loaded_uav | rt_loaded_heap)) return;
    rt_report_frame = frame_number;
    log("x2w-draw-rt totals: frame %ld; loaded bufword3 %ld uav %ld heap %ld; bufword3 values set %ld zero %ld; "
        "uav draws %ld, buffer uavs bound %ld null (unresolved %ld misaligned %ld); srvs in place %ld stale copy %ld; "
        "texture uavs bound %ld, draws skipped (unresolved %ld flag %ld shape %ld format %ld store %ld); heap draws %ld skipped %ld",
        frame_number, rt_loaded_word3, rt_loaded_uav, rt_loaded_heap, rt_word3_set, rt_word3_zero,
        rt_uav_draws, rt_uav_buf_bound, rt_uav_buf_unresolved, rt_uav_buf_misaligned, rt_srv_in_place, rt_srv_stale_copy,
        rt_uav_tex[0], rt_uav_tex[1], rt_uav_tex[2], rt_uav_tex[3], rt_uav_tex[4], rt_uav_tex[5], rt_heap_draws, rt_heap_skipped);
}

bool texture_census_on();
bool census_every_event();
void report_memory_srv_census(long frame);
enum TextureSource { ts_linear_srv, ts_tiled_address, ts_other, ts_detiled, ts_count };
LONG64 census_real[ts_count], census_null[x2w_local::null_cause_count], census_real_format[128];
LONG census_report_frame = -1;
void log_texture_census(const char* when);
void census_texture_binding(Wrapper* resource, ID3D12Resource* texture, x2w_local::NullCause cause)
{
    if (!texture_census_on()) return;
    if (!texture) {
        InterlockedIncrement64(&census_null[static_cast<int>(cause)]);
    } else {
        const int source = resource && resource->own().detiled_memory_srv ? ts_detiled
                         : resource && resource->own().linear_memory_srv ? ts_linear_srv
                         : resource && resource->own().tiled_address_srv ? ts_tiled_address : ts_other;
        InterlockedIncrement64(&census_real[source]);
        InterlockedIncrement64(&census_real_format[static_cast<UINT>(texture->GetDesc().Format) & 127]);
    }
    if (census_every_event()) log_texture_census("binding");
}
void report_texture_census()  // draw_lock held
{
    if (!texture_census_on() || frame_number % 300 || frame_number == census_report_frame) return;
    census_report_frame = frame_number;
    char when[32];
    std::snprintf(when, sizeof(when), "frame %ld", frame_number);
    log_texture_census(when);
    report_memory_srv_census(frame_number);
}
void log_texture_census(const char* when)
{
    char formats[512] = "";
    int used = 0;
    for (int f = 0; f < 128 && used < 480; ++f)
        if (census_real_format[f]) used += std::snprintf(formats + used, sizeof(formats) - used, " %d:%lld", f, census_real_format[f]);
    char nulls[256] = "";
    used = 0;
    for (int c = 0; c < x2w_local::null_cause_count && used < 230; ++c)
        if (census_null[c]) used += std::snprintf(nulls + used, sizeof(nulls) - used, " cause%d:%lld", c, census_null[c]);
    log("x2w-texcensus draw textures %s (cumulative): real %lld (linear memory srv %lld, tiled address %lld, other %lld, detiled %lld), "
        "null%s; real by format%s",
        when, census_real[0] + census_real[1] + census_real[2] + census_real[3], census_real[0], census_real[1], census_real[2],
        census_real[3], nulls, formats);
}

void count_null_binding(x2w_local::NullCause cause)
{
    if (cause == x2w_local::NullCause::none) return;
    InterlockedIncrement(&null_bindings);
    InterlockedIncrement(&null_by_cause[static_cast<int>(cause)]);
}
struct WatchedSlot { u64 at, key, slot; UINT srv; std::uint32_t words[8]; LONG since; bool reported; };
SRWLOCK watch_lock = SRWLOCK_INIT;
std::vector<WatchedSlot>& watched_slots() { static std::vector<WatchedSlot> v; return v; }
extern LONG frame_number;
void watch_descriptor(u64 at, const std::uint32_t* words, u64 key, UINT srv, u64 slot)
{
    AcquireSRWLockExclusive(&watch_lock);
    if (watched_slots().size() < 64) {
        WatchedSlot w{at, key, slot, srv, {}, frame_number, false};
        std::memcpy(w.words, words, sizeof(w.words));
        watched_slots().push_back(w);
    }
    ReleaseSRWLockExclusive(&watch_lock);
}
void note_descriptor_watch(LONG frame)
{
    AcquireSRWLockExclusive(&watch_lock);
    for (WatchedSlot& w : watched_slots()) {
        if (w.reported) continue;
        std::uint32_t now[8]{};
        if (!x2w_local::readable(reinterpret_cast<const void*>(w.at), sizeof(now))) continue;
        std::memcpy(now, reinterpret_cast<const void*>(w.at), sizeof(now));
        if (!std::memcmp(now, w.words, sizeof(now))) continue;
        w.reported = true;
        log("x2w-draw watched slot 0x%llx (for %016llx srv %u, watched since frame %ld) changed at frame %ld: %08x %08x %08x %08x %08x %08x %08x %08x",
            w.slot, w.key, w.srv, w.since, frame, now[0], now[1], now[2], now[3], now[4], now[5], now[6], now[7]);
    }
    ReleaseSRWLockExclusive(&watch_lock);
}

bool refused_marker_description(const std::uint32_t* w, char* out, std::size_t size);  // below
void note_null_texture(u64 key, UINT srv, u64 entry, const std::uint32_t* words, x2w_local::NullCause cause)
{
    static const bool on = [] {
        const bool value = x2w_local::env_on("X2W_DRAW_NULL_TEXTURE_LOG");
        if (value) log("x2w-draw X2W_DRAW_NULL_TEXTURE_LOG=1: null texture bindings are logged per pipeline key and srv");
        return value;
    }();
    if (!on || cause == x2w_local::NullCause::none) return;
    static SRWLOCK lock = SRWLOCK_INIT;
    static std::unordered_set<u64>* seen = new std::unordered_set<u64>;
    static LONG lines;
    AcquireSRWLockExclusive(&lock);
    const bool fresh = seen->insert(key * 31 + srv).second;
    ReleaseSRWLockExclusive(&lock);
    if (!fresh || !first_calls(lines, 96)) return;
    log("x2w-draw null texture %016llx srv %u cause %s frame %ld: entry 0x%llx words %08x %08x %08x %08x %08x %08x %08x %08x", key, srv,
        x2w_local::null_cause_names[static_cast<int>(cause)], frame_number, entry, words[0], words[1], words[2], words[3], words[4],
        words[5], words[6], words[7]);
    if (char marker[256]; refused_marker_description(words, marker, sizeof marker)) {
        log("x2w-draw null texture %016llx srv %u: %s", key, srv, marker);
        return;
    }
    bool rest_zero = true;
    for (int i = 1; i < 8; ++i) rest_zero = rest_zero && words[i] == 0;
    const bool link = words[0] && words[0] != view_token_magic && rest_zero;
    // A token whose first word the title replaced (a freed slot that still names a view).
    const bool clobbered = words[0] != view_token_magic && words[2] >= 17 && words[2] <= 22 && (words[4] || words[5]);
    if (!link && !clobbered) return;
    AcquireSRWLockShared(&note_lock);
    const std::vector<HeapInfo> heaps = heap_infos();
    ViewRecord named{};
    const bool named_known = clobbered && words[1] < view_records().size();
    if (named_known) named = view_records()[words[1]];
    ReleaseSRWLockShared(&note_lock);
    if (clobbered)
        log("x2w-draw null texture %016llx srv %u: looks like a freed token: first word 0x%x, view id %u slot %u resource 0x%08x%08x; view %s%s",
            key, srv, words[0], words[1], words[2], words[5], words[4], named_known ? "known" : "UNKNOWN",
            named_known && named.resource == (u64{words[5]} << 32 | words[4]) ? ", resource matches" : named_known ? ", resource DIFFERS" : "");
    bool in_heap = false;
    for (std::size_t h = 0; h < heaps.size(); ++h) {
        const HeapInfo& heap = heaps[h];
        if (!heap.title_cpu) continue;
        const UINT stride = title_descriptor_stride(heap.title_cpu, heap.increment);
        if (entry < heap.title_cpu || entry >= heap.title_cpu + u64{heap.count} * stride) continue;
        in_heap = true;
        const u64 index = (entry - heap.title_cpu) / stride;
        log("x2w-draw null texture %016llx srv %u: entry 0x%llx is index 0x%llx of heap %zu (type %u, count %u, title 0x%llx, stride %u); shadow id %u",
            key, srv, entry, index, h, heap.type, heap.count, heap.title_cpu, stride, heap.shadow ? heap.shadow[index] : 0u);
        if (!link) break;
        for (int d = -1; d <= 0; ++d) {
            const u64 slot = u64{words[0]} + d;
            if (slot >= heap.count) continue;
            const u64 at = heap.title_cpu + slot * stride;
            std::uint32_t there[8]{};
            const bool readable = x2w_local::readable(reinterpret_cast<const void*>(at), sizeof(there));
            if (readable) std::memcpy(there, reinterpret_cast<const void*>(at), sizeof(there));
            const std::uint32_t shadow = heap.shadow ? heap.shadow[slot] : 0;
            log("x2w-draw null texture %016llx srv %u: heap %zu slot 0x%llx (N%s): %s words %08x %08x %08x %08x %08x %08x; shadow id %u%s",
                key, srv, h, slot, d ? "-1" : "", readable ? (there[0] == view_token_magic ? "token" : "no token") : "unreadable", there[0], there[1],
                there[2], there[3], there[4], there[5], shadow, shadow ? "" : " (no view ever created there)");
            if (d == -1 && readable) watch_descriptor(at, there, key, srv, slot);
        }
        break;
    }
    if (!in_heap) log("x2w-draw null texture %016llx srv %u: entry 0x%llx is in no title descriptor heap", key, srv, entry);
}
LONG frame_number;

LONG heap_index_checks, heap_index_restored, heap_index_slot_token, heap_index_shadow, heap_index_unresolved;
void resolve_heap_index(u64 key, UINT srv, u64 entry, std::uint32_t* words)
{
    static const bool on = [] {
        const bool value = x2w_local::env_on("X2W_DRAW_TEXTURE_HEAP_INDEX");
        if (value) log("x2w-draw X2W_DRAW_TEXTURE_HEAP_INDEX=1: freed or free-slot texture descriptors are resolved (slot N-1 / view id)");
        return value;
    }();
    if (!on || words[0] == view_token_magic) return;
    bool rest_zero = true;
    for (int i = 1; i < 8; ++i) rest_zero = rest_zero && words[i] == 0;
    const bool link = words[0] && rest_zero;
    const bool clobbered = words[2] >= 17 && words[2] <= 22 && (words[4] || words[5]);
    if (!link && !clobbered) return;
    const LONG checks = InterlockedIncrement(&heap_index_checks);
    std::uint32_t out[8]{};
    const char* how = nullptr;
    AcquireSRWLockShared(&note_lock);
    if (clobbered) {
        if (words[1] < view_records().size() && view_records()[words[1]].resource == (u64{words[5]} << 32 | words[4])) {
            std::memcpy(out, words, sizeof(out));
            out[0] = view_token_magic;
            how = "restored token";
        }
    } else {
        for (const HeapInfo& heap : heap_infos()) {
            if (!heap.title_cpu || words[0] > heap.count) continue;
            const UINT stride = title_descriptor_stride(heap.title_cpu, heap.increment);
            if (entry < heap.title_cpu || entry >= heap.title_cpu + u64{heap.count} * stride) continue;
            const u64 slot = u64{words[0]} - 1;
            const u64 at = heap.title_cpu + slot * stride;
            if (x2w_local::readable(reinterpret_cast<const void*>(at), sizeof(out))) {
                std::memcpy(out, reinterpret_cast<const void*>(at), sizeof(out));
                if (out[0] == view_token_magic && out[1] < view_records().size()) how = "slot N-1 token";
            }
            const std::uint32_t shadow = heap.shadow ? heap.shadow[slot] : 0;
            if (!how && shadow && shadow - 1 < view_records().size()) {
                const ViewRecord& r = view_records()[shadow - 1];
                const std::uint32_t token[8] = {view_token_magic, shadow - 1, r.slot, 0, static_cast<std::uint32_t>(r.resource),
                                                static_cast<std::uint32_t>(r.resource >> 32), 0, 0};
                std::memcpy(out, token, sizeof(out));
                how = "slot N-1 shadow id";
            }
            break;
        }
    }
    ReleaseSRWLockShared(&note_lock);
    InterlockedIncrement(!how ? &heap_index_unresolved : how[0] == 'r' ? &heap_index_restored : how[9] == 't' ? &heap_index_slot_token
                                                                                               : &heap_index_shadow);
    static LONG reports;
    if ((checks & 4095) == 0 && first_calls(reports, 40))
        log("x2w-draw texture heap index totals: checks %ld, restored tokens %ld, slot N-1 tokens %ld, slot N-1 shadow ids %ld, unresolved %ld", checks,
            heap_index_restored, heap_index_slot_token, heap_index_shadow, heap_index_unresolved);
    static SRWLOCK lock = SRWLOCK_INIT;
    static std::unordered_set<u64>* seen = new std::unordered_set<u64>;
    AcquireSRWLockExclusive(&lock);
    const bool fresh = seen->size() < 4096 && seen->insert(key * 31 + srv).second;
    ReleaseSRWLockExclusive(&lock);
    static LONG lines;
    if (fresh && first_calls(lines, 48))
        log("x2w-draw texture heap index %016llx srv %u frame %ld: entry 0x%llx words %08x %08x %08x -> %s%s (view id %u slot %u)", key, srv,
            frame_number, entry, words[0], words[1], words[2], how ? how : "unresolved", how ? "" : ", left null", how ? out[1] : 0u, how ? out[2] : 0u);
    if (how) std::memcpy(words, out, sizeof(out));
}

bool buffer_heap_index_on()
{
    static const bool on = [] {
        const bool v = x2w_local::env_on("X2W_BUFFER_HEAP_INDEX");
        if (v) log("x2w-draw X2W_BUFFER_HEAP_INDEX=1: slot-70 buffers without a resource get a stand-in; free-list buffer entries bind null");
        return v;
    }();
    return on;
}
LONG buffer_links, buffer_links_nulled, buffer_refused_views;
// With the switch, a slot-70 call that is refused writes this marker and its reason's code into its destination
// (mark_refused_destination, d3d12_memory_srv.inl), so an entry copied from it names the refusal instead of a link.
constexpr std::uint32_t refused_view_magic = 0x52463258;  // "X2FR"
SRWLOCK refusal_names_lock = SRWLOCK_INIT;
const char* refusal_names[64];
std::uint32_t refusal_code(const char* reason)
{
    AcquireSRWLockExclusive(&refusal_names_lock);
    std::uint32_t code = 0;
    while (code < 63 && refusal_names[code] && std::strcmp(refusal_names[code], reason)) ++code;
    if (!refusal_names[code]) refusal_names[code] = reason;  // reasons are string literals
    ReleaseSRWLockExclusive(&refusal_names_lock);
    return code;
}
const char* refusal_name(std::uint32_t code)
{
    AcquireSRWLockShared(&refusal_names_lock);
    const char* name = code < 64 && refusal_names[code] ? refusal_names[code] : "unknown";
    ReleaseSRWLockShared(&refusal_names_lock);
    return name;
}
// Why X2W_MEMORY_SRV_DETILE did not take a refused texture call (d3d12_memory_srv.inl detile_decline); 0 = not asked
// (switch off, or not a texture description).
const char* const detile_declines[] = {"-", "not console-swizzled", "a tiled path's form", "announced", "view form",
    "layout or element size", "(6: bc mip chain, retired)", "over 1 GiB", "unreadable", "destination", "publish failed", "would detile"};
// The marker's description words (mark_refused_destination): word 2 = bit 31 | format | layout << 8 | dimension << 17 |
// min(mips, 15) << 20 | detile decline << 24, word 3 = width, word 4 = height, word 5 = depth or array size | first 256
// bytes readable << 16 | nonzero << 17, words 6/7 = the address. Bit 31 keeps word 2 out of the token slot range.
void describe_marker(const std::uint32_t* w, char* out, std::size_t size)
{
    if (!(w[2] >> 31)) {
        std::snprintf(out, size, "no description");
        return;
    }
    const unsigned decline = (w[2] >> 24) & 15;
    std::snprintf(out, size, "fmt %u layout 0x%x dim %u mips %u %ux%ux%u address 0x%llx %s, detile: %s", w[2] & 0xff, (w[2] >> 8) & 0x1ff,
                  (w[2] >> 17) & 7, (w[2] >> 20) & 15, w[3], w[4], w[5] & 0xffff, u64{w[7]} << 32 | w[6],
                  !((w[5] >> 16) & 1) ? "unreadable" : (w[5] >> 17) & 1 ? "nonzero" : "zero",
                  decline < std::size(detile_declines) ? detile_declines[decline] : "?");
}
bool refused_marker_description(const std::uint32_t* w, char* out, std::size_t size)
{
    if (w[0] != refused_view_magic) return false;
    char d[160];
    describe_marker(w, d, sizeof d);
    std::snprintf(out, size, "refused: %s; %s", refusal_name(w[1]), d);
    return true;
}
LONG marker_rewrites;
bool marker_rewrite_on()
{
    static const bool on = [] {
        const bool v = x2w_local::env_on("X2W_MARKER_REWRITE_LOG");
        if (v) log("x2w-draw X2W_MARKER_REWRITE_LOG=1: table entries holding a refused-view marker whose first word was rewritten are counted and logged");
        return v;
    }();
    return on;
}
// The description's fixed shape (describe_marker): bit 31 set, bits 28-30 clear, a decline in range, a dimension
// 1-4, a nonzero width, word 5 bits 18-31 clear; and word 0 neither the marker nor a token.
bool rewritten_marker_words(const std::uint32_t* w)
{
    const unsigned dim = (w[2] >> 17) & 7;
    return w[0] != refused_view_magic && w[0] != view_token_magic && (w[2] >> 31) && !((w[2] >> 28) & 7) &&
           ((w[2] >> 24) & 15) < std::size(detile_declines) && dim >= 1 && dim <= 4 && w[3] && !(w[5] >> 18);
}
void note_rewritten_marker(u64 key, UINT srv, u64 entry, const std::uint32_t* w)
{
    if (!marker_rewrite_on() || !rewritten_marker_words(w)) return;
    const LONG n = InterlockedIncrement(&marker_rewrites);
    if ((n & (n - 1)) == 0) log("x2w-draw rewritten refused-view markers %ld", n);
    static LONG shown;
    if (!first_calls(shown, 8)) return;
    char d[160];
    describe_marker(w, d, sizeof d);
    log("x2w-draw rewritten refused-view marker %016llx srv %u frame %ld: entry 0x%llx words %08x %08x %08x %08x %08x %08x %08x %08x; "
        "described as %s; reason code %u if word 1 is intact; base as a T# 0x%llx, as a V# 0x%llx", key, srv, frame_number, entry, w[0], w[1], w[2],
        w[3], w[4], w[5], w[6], w[7], d, w[1], (u64{w[0]} | u64{w[1] & 0xff} << 32) << 8, u64{w[0]} | u64{w[1] & 0xffff} << 32);
}
bool free_link_words(const std::uint32_t* w)
{
    if (w[0] == refused_view_magic) return true;
    if (!w[0] || w[0] == view_token_magic || w[3]) return false;
    return (!w[1] && !w[2]) || (w[2] >= 17 && w[2] <= 22);
}
void log_buffer_link(u64 key, UINT srv, u64 entry, const std::uint32_t* words)
{
    AcquireSRWLockShared(&note_lock);
    const std::vector<HeapInfo> heaps = heap_infos();
    ReleaseSRWLockShared(&note_lock);
    char where[96] = "in no title descriptor heap";
    char slots[2][160] = {"", ""};
    for (std::size_t h = 0; h < heaps.size(); ++h) {
        const HeapInfo& heap = heaps[h];
        if (!heap.title_cpu) continue;
        const UINT stride = title_descriptor_stride(heap.title_cpu, heap.increment);
        if (entry < heap.title_cpu || entry >= heap.title_cpu + u64{heap.count} * stride) continue;
        std::snprintf(where, sizeof where, "index 0x%llx of heap %zu (count %u)", (entry - heap.title_cpu) / stride, h, heap.count);
        for (int d = -1; d <= 0; ++d) {
            const u64 slot = u64{words[0]} + d;
            if (slot >= heap.count) continue;
            const u64 at = heap.title_cpu + slot * stride;
            std::uint32_t there[6]{};
            const bool readable = x2w_local::readable(reinterpret_cast<const void*>(at), sizeof(there));
            if (readable) std::memcpy(there, reinterpret_cast<const void*>(at), sizeof(there));
            std::snprintf(slots[d + 1], sizeof slots[0], "slot %s 0x%llx %s %08x %08x %08x %08x %08x %08x shadow %u", d ? "N-1" : "N", slot,
                          readable ? (there[0] == view_token_magic ? "token" : "words") : "unreadable", there[0], there[1], there[2], there[3],
                          there[4], there[5], heap.shadow ? heap.shadow[slot] : 0u);
        }
        break;
    }
    char kind[80], described[192] = "";
    if (words[0] == refused_view_magic) {
        std::snprintf(kind, sizeof kind, "refused: %s", refusal_name(words[1]));
        char d[160];
        describe_marker(words, d, sizeof d);
        std::snprintf(described, sizeof described, "; refused view: %s", d);
    } else {
        std::snprintf(kind, sizeof kind, "%s", words[1] || words[2] ? "freed token" : "free link");
    }
    log("x2w-frame srv-heapbuf %016llx %u frame %ld: entry 0x%llx words %08x %08x %08x %08x (%s) is %s; %s; %s; bound null%s", key, srv, frame_number,
        entry, words[0], words[1], words[2], words[3], kind, where, slots[0], slots[1], described);
}

u64 fnv_bytes(const void* data, std::size_t bytes, u64 hash = 1469598103934665603ull)
{
    for (std::size_t i = 0; i < bytes; ++i) hash = (hash ^ static_cast<const unsigned char*>(data)[i]) * 1099511628211ull;
    return hash;
}

bool cache_path(wchar_t* out, const wchar_t* prefix, u64 key, const wchar_t* extension)
{
    wchar_t dir[MAX_PATH];
    const DWORD n = GetEnvironmentVariableW(L"X2W_SHADER_CACHE", dir, MAX_PATH);
    if (!n || n >= MAX_PATH) return false;
    swprintf(out, MAX_PATH, L"%s\\%s%016llx%s", dir, prefix, key, extension);
    return true;
}

bool read_text(const wchar_t* path, std::string& out)
{
    HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    char buffer[8192];
    DWORD got = 0;
    while (ReadFile(file, buffer, sizeof(buffer), &got, nullptr) && got) out.append(buffer, got);
    CloseHandle(file);
    return true;
}

// Publish each input atomically. The callers write the request's ready marker last.
bool publish_shader_request(u64 key, const wchar_t* ext, const void* data, std::size_t bytes)
{
    wchar_t path[MAX_PATH], temporary[MAX_PATH], suffix[32];
    if (!cache_path(path, L"req_", key, ext)) return false;
    if (GetFileAttributesW(path) != INVALID_FILE_ATTRIBUTES) return true;
    swprintf(suffix, std::size(suffix), L"%s.%lu.tmp", ext, GetCurrentProcessId());
    if (!cache_path(temporary, L"req_", key, suffix)) return false;
    HANDLE file = CreateFileW(temporary, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    DWORD written = 0;
    const bool complete = WriteFile(file, data, static_cast<DWORD>(bytes), &written, nullptr) && written == bytes;
    CloseHandle(file);
    if (complete && MoveFileExW(temporary, path, 0)) return true;
    DeleteFileW(temporary);
    return GetFileAttributesW(path) != INVALID_FILE_ATTRIBUTES;
}

HRESULT create_draw_root(Translation& tr)
{
    const UINT n = static_cast<UINT>(tr.srvs.size()), m = static_cast<UINT>(tr.uavs.size());
    if (!m && !tr.heap)
        return x2w_startup::create_translated_root(device_of(the_device), static_cast<UINT>(tr.uds.size()), n, tr.samplers, &tr.root);
    D3D12_ROOT_PARAMETER parameters[9]{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters[0].Constants.Num32BitValues = static_cast<UINT>(tr.uds.size());
    const D3D12_DESCRIPTOR_RANGE srv_range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, n, 0, 0, 0};
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[1].DescriptorTable = {1, &srv_range};
    const D3D12_DESCRIPTOR_RANGE sampler_range{D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, x2w_startup::translated_sampler_count, 0,
                                               x2w_startup::translated_sampler_space, 0};
    UINT count = x2w_startup::sampler_root_parameter(n);
    if (tr.samplers.heap) {
        parameters[count].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[count].DescriptorTable = {1, &sampler_range};
        ++count;
    }
    const D3D12_DESCRIPTOR_RANGE uav_range{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, m, 0, 0, 0};
    if (m) {
        tr.uav_parameter = count;
        parameters[count].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[count].DescriptorTable = {1, &uav_range};
        ++count;
    }
    D3D12_DESCRIPTOR_RANGE heap_ranges[5]{};
    const D3D12_DESCRIPTOR_RANGE bounded_ranges[3] = {{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 1, 0}, {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1, 1, 0},
                                                      {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2, 1, 0}};
    if (tr.heap) {
        tr.heap_parameter = count;
        parameters[count].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[count].DescriptorTable = {1, &bounded_ranges[0]};
        parameters[count + 1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[count + 1].DescriptorTable = {1, &bounded_ranges[1]};
        for (UINT k = 0; k < 5; ++k) heap_ranges[k] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, UINT_MAX, 0, 2 + k, 0};
        parameters[count + 2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[count + 2].DescriptorTable = {5, heap_ranges};
        count += 3;
    }
    if (tr.heap && tr.heapwords) {
        tr.vdesc_parameter = count;
        parameters[count].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[count++].DescriptorTable = {1, &bounded_ranges[2]};
    }
    const auto statics = x2w_startup::decode_statics(tr.samplers);
    const D3D12_ROOT_SIGNATURE_DESC desc{count, parameters, static_cast<UINT>(statics.size()), statics.data(), D3D12_ROOT_SIGNATURE_FLAG_NONE};
    x2w_startup::ComPtr<ID3DBlob> blob, errors;
    HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &blob, &errors);
    if (FAILED(hr) && errors) log("x2w-draw-rt %016llx root signature: %.300s", tr.key, static_cast<const char*>(errors->GetBufferPointer()));
    if (SUCCEEDED(hr)) hr = device_of(the_device)->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&tr.root));
    return hr;
}

struct ManifestWatch {
    SRWLOCK lock = SRWLOCK_INIT;
    std::vector<std::pair<Translation*, std::string>> missing;  // translation, its rtv request
};
ManifestWatch& manifest_watch() { static ManifestWatch w; return w; }
std::unordered_map<const Translation*, volatile LONG>& manifest_seen() { static std::unordered_map<const Translation*, volatile LONG> m; return m; }
bool manifest_poll() { static const bool on = x2w_local::env_on("X2W_MANIFEST_POLL"); return on; }

void manifest_check(const wchar_t* dir, const wchar_t* name)
{
    std::vector<std::pair<Translation*, std::string>> look;
    AcquireSRWLockShared(&manifest_watch().lock);
    look = manifest_watch().missing;
    ReleaseSRWLockShared(&manifest_watch().lock);
    if (look.empty() || (!name && !dir[0])) return;
    std::unordered_set<std::wstring> present;
    if (!name) {
        std::wstring pattern = std::wstring(dir) + L"\\*.txt";
        WIN32_FIND_DATAW found{};
        const HANDLE find = FindFirstFileExW(pattern.c_str(), FindExInfoBasic, &found, FindExSearchNameMatch, nullptr,
                                             FIND_FIRST_EX_LARGE_FETCH);
        if (find == INVALID_HANDLE_VALUE) return;
        do {
            std::wstring file(found.cFileName);
            for (wchar_t& c : file) c = static_cast<wchar_t>(towlower(c));
            present.insert(std::move(file));
        } while (FindNextFileW(find, &found));
        FindClose(find);
    }
    for (const auto& [tr, request] : look) {
        wchar_t path[MAX_PATH];
        if (!cache_path(path, L"", tr->key, L".txt")) continue;
        const wchar_t* slash = std::wcsrchr(path, L'\\');
        std::wstring file(slash ? slash + 1 : path);
        for (wchar_t& c : file) c = static_cast<wchar_t>(towlower(c));
        if (name ? _wcsicmp(file.c_str(), name) != 0 : !present.count(file)) continue;
        AcquireSRWLockExclusive(&manifest_watch().lock);
        auto& m = manifest_watch().missing;
        m.erase(std::remove_if(m.begin(), m.end(), [&](const auto& e) { return e.first == tr; }), m.end());
        InterlockedExchange(&manifest_seen()[tr], 1);
        ReleaseSRWLockExclusive(&manifest_watch().lock);
        log("x2w-draw manifest for %016llx appeared: reloaded at its next draw", tr->key);
    }
}

DWORD WINAPI manifest_watcher(void*)
{
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    wchar_t dir[MAX_PATH] = L"";
    HANDLE handle = INVALID_HANDLE_VALUE;
    if (cache_path(dir, L"", 0, L".txt")) {
        if (wchar_t* slash = std::wcsrchr(dir, L'\\')) *slash = 0;
        handle = CreateFileW(dir, FILE_LIST_DIRECTORY, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                             OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED, nullptr);
    }
    log("x2w-draw manifest watcher: %s", handle != INVALID_HANDLE_VALUE ? "directory change records" : "30 s rescans");
    alignas(DWORD) static unsigned char buffer[64 * 1024];
    OVERLAPPED overlapped{};
    overlapped.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    bool pending = false;
    bool scan = true;  // first pass: a manifest published before the watch started
    for (;;) {
        // File names only: the stand-in itself writes req_* files here; a manifest is published by rename
        // or creation, both name changes, and the record names the file, so no query is needed. The read
        // is armed before any scan, so a file named during the scan is still recorded.
        if (!pending && handle != INVALID_HANDLE_VALUE && overlapped.hEvent) {
            ResetEvent(overlapped.hEvent);
            pending = ReadDirectoryChangesW(handle, buffer, sizeof(buffer), FALSE, FILE_NOTIFY_CHANGE_FILE_NAME, nullptr,
                                            &overlapped, nullptr) != FALSE;
        }
        if (scan) manifest_check(dir, nullptr);
        scan = true;
        // 30 s rescan: a manifest published between a failed open and its registration has no record.
        if (!pending) {
            Sleep(30000);
            continue;
        }
        if (WaitForSingleObject(overlapped.hEvent, 30000) != WAIT_OBJECT_0) continue;
        pending = false;
        DWORD bytes = 0;
        if (!GetOverlappedResult(handle, &overlapped, &bytes, FALSE) || bytes == 0) continue;  // overflow: names lost
        scan = false;
        for (DWORD at = 0;;) {
            const auto* record = reinterpret_cast<const FILE_NOTIFY_INFORMATION*>(buffer + at);
            if (record->Action == FILE_ACTION_ADDED || record->Action == FILE_ACTION_RENAMED_NEW_NAME) {
                wchar_t name[MAX_PATH];
                const std::size_t chars = record->FileNameLength / sizeof(wchar_t);
                if (chars < MAX_PATH) {
                    std::wmemcpy(name, record->FileName, chars);
                    name[chars] = 0;
                    if (chars > 4 && _wcsicmp(name + chars - 4, L".txt") == 0) manifest_check(dir, name);
                }
            }
            if (!record->NextEntryOffset) break;
            at += record->NextEntryOffset;
        }
    }
}

void watch_missing_manifest(Translation& tr, const std::string& request)
{
    static const HANDLE thread = CreateThread(nullptr, 0, manifest_watcher, nullptr, 0, nullptr);
    (void)thread;
    AcquireSRWLockExclusive(&manifest_watch().lock);
    manifest_watch().missing.emplace_back(&tr, request);
    manifest_seen()[&tr] = 0;
    ReleaseSRWLockExclusive(&manifest_watch().lock);
}

// draw_lock held: true when the translation's absent manifest should be read again now.
bool manifest_retry_due(Translation& tr)
{
    if (!tr.retry_manifest_at) return false;
    if (manifest_poll()) return GetTickCount64() >= tr.retry_manifest_at;
    AcquireSRWLockShared(&manifest_watch().lock);
    const auto found = manifest_seen().find(&tr);
    const bool seen = found != manifest_seen().end() && InterlockedExchange(&found->second, 0) != 0;
    ReleaseSRWLockShared(&manifest_watch().lock);
    return seen;
}

void load_translation(Translation& tr, const std::string& rtv_request = {})
{
    tr.state = 2;
    wchar_t path[MAX_PATH];
    std::string manifest;
    if (!cache_path(path, L"", tr.key, L".txt")) return;
    if (!read_text(path, manifest)) {
        if (!tr.retry_manifest_at) {
            InterlockedIncrement(&translation_missing);
            static LONG missing;
            if (rtv_request.empty() && first_calls(missing, 12))
                log("x2w-draw no translation for pipeline %016llx", tr.key);
        }
        tr.retry_manifest_at = GetTickCount64() + 1000;
        if (!manifest_poll()) watch_missing_manifest(tr, rtv_request);
        return;
    }
    // The writer publishes this marker last. A malformed published result stays rejected.
    const bool previously_missing = tr.retry_manifest_at != 0;
    tr.retry_manifest_at = 0;
    bool matched_rtv = false;
    for (std::size_t at = 0; at < manifest.size();) {
        std::size_t end = manifest.find('\n', at);
        if (end == std::string::npos) end = manifest.size();
        std::string line = manifest.substr(at, end - at);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        at = end + 1;
        int i = 0, x = 0, y = 0;
        char kind[16] = "", source[16] = "";
        if (line.compare(0, 5, "rtv1 ") == 0) {
            if (matched_rtv || line != rtv_request) {
                log("x2w-draw RTV manifest mismatch for %016llx", tr.key);
                return;
            }
            matched_rtv = true;
        } else if (line.compare(0, 13, "x2w-manifest ") == 0 && line != "x2w-manifest 1") {
            // Version 2 adds `uav` lines for vertex-stage stores. This runtime does not bind them,
            // so it must not draw a translation whose stores it would silently skip (#1723).
            log("x2w-draw unsupported manifest version for %016llx", tr.key);
            return;
        } else if (std::sscanf(line.c_str(), "udsrc %d %15s %d %d", &i, source, &x, &y) >= 2) {
            Translation::Ud ud{-1, x, y};
            if (!std::strcmp(source, "rect")) ud.kind = Translation::rect;
            else if (!std::strcmp(source, "basevertex")) ud.kind = Translation::base_vertex;
            else if (!std::strcmp(source, "startinstance")) ud.kind = Translation::start_instance;
            else if (!std::strcmp(source, "rootconst")) ud.kind = Translation::root_constant;
            else if (!std::strcmp(source, "stride")) ud.kind = Translation::stride;
            else if (!std::strcmp(source, "d16unorm16") && x >= 0 && x < 32) ud.kind = Translation::d16_unorm16;
            else if (!std::strcmp(source, "descword3")) ud.kind = Translation::desc_word3;
            else if (!std::strcmp(source, "mrtbits") && !rtv_request.empty()) ud.kind = Translation::mrt_bits;
            else if (!std::strcmp(source, "bufword3")) ud.kind = Translation::buf_word3;  // word 3 of srv a's descriptor
            else if (!std::strcmp(source, "uavword3")) ud.kind = Translation::uav_word3;  // word 3 of uav a's descriptor
            else if (!std::strcmp(source, "uavstride") && x >= 0 && x < 8) ud.kind = Translation::uav_stride;
            else if (!std::strcmp(source, "uavbase") && x >= 0 && x < 8) {
                ud.kind = Translation::uav_base;
                tr.uav_base_wanted |= 1u << x;
            } else if (!std::strcmp(source, "descword")) {
                static const bool descword_on = x2w_local::env_on("X2W_MESH_DESCWORD");
                if (!descword_on || x < 0 || x >= 16 || y < 0) {
                    static LONG logged;
                    if (first_calls(logged, 8))
                        log("x2w-draw %016llx: `udsrc %d descword %d %d` %s: translation refused", tr.key, i, x, y,
                            descword_on ? "out of range" : "needs X2W_MESH_DESCWORD=1");
                    return;
                }
                ud.kind = Translation::desc_word;
            }
            if (static_cast<std::size_t>(i) == tr.uds.size() && ud.kind >= 0) tr.uds.push_back(ud);
        } else if (std::sscanf(line.c_str(), "uav %d %15s %15s %d %d", &i, kind, source, &x, &y) >= 4) {
            const int uav_kind = !std::strcmp(kind, "buf") ? 0 : !std::strcmp(kind, "tex2d") && !std::strcmp(source, "table") ? 1 : -1;
            const int uav_source = !std::strcmp(source, "table") ? 0 : !std::strcmp(source, "root") ? 2 : -1;
            if (uav_kind < 0 || uav_source < 0 || static_cast<std::size_t>(i) != tr.uavs.size() || tr.uavs.size() >= 8) {
                log("x2w-draw-rt %016llx: unsupported manifest line \"%s\"", tr.key, line.c_str());
                return;
            }
            tr.uavs.push_back({uav_kind, uav_source, x, y});
        } else if (std::sscanf(line.c_str(), "srv %d %15s %15s %d %d", &i, kind, source, &x, &y) >= 4) {
            const int texture_kind = !std::strcmp(kind, "tex2d") ? 1 : !std::strcmp(kind, "tex2da") ? 2 :
                                     !std::strcmp(kind, "tex3d") ? 3 : !std::strcmp(kind, "tex1d") ? 4 : 0;
            Translation::Srv srv{texture_kind, !std::strcmp(source, "vb") ? 1 : !std::strcmp(source, "root") ? 2 : 0, x, y};
            if (static_cast<std::size_t>(i) == tr.srvs.size()) tr.srvs.push_back(srv);
        } else if (line == "heap 1") {
            tr.heap = true;
        } else if (line == "heapwords 1") {
            tr.heapwords = true;
        } else if (line.compare(0, 6, "wmask ") == 0) {
            if (!tr.write_masks.read_manifest_line(line)) {
                log("x2w-draw invalid write mask manifest line for %016llx: %s", tr.key, line.c_str());
                return;
            }
        } else if (line.compare(0, 7, "d16srv ") == 0) {
            if (!tr.d16_srvs.read_manifest_line(line)) {
                log("x2w-draw invalid d16srv manifest line for %016llx: %s", tr.key, line.c_str());
                return;
            }
        } else if (line.compare(0, 4, "uav ") == 0) {
            // tools/startup/gfx2hlsl.py lowers a pixel-stage image_store to a u-register store, but the
            // translated root signature has no UAV range and no view is made from the title's descriptor yet.
            log("x2w-draw pipeline %016llx stores to an image (%s); pixel-stage UAVs are not bound yet, translation not used",
                tr.key, line.c_str());
            return;
        } else if (!tr.samplers.read_manifest_line(line.c_str())) {
            log("x2w-draw invalid sampler heap manifest for %016llx", tr.key);
            return;
        }
    }
    if (matched_rtv != !rtv_request.empty()) {
        log("x2w-draw RTV manifest missing for %016llx", tr.key);
        return;
    }
    if (tr.uds.empty() || tr.uds.size() > 32 || tr.srvs.size() > draw_srv_limit || !tr.d16_srvs.valid_for_srvs(tr.srvs)) return;
    for (const Translation::Ud& ud : tr.uds) {
        if (ud.kind == Translation::d16_unorm16 &&
            (ud.a < 0 || static_cast<std::size_t>(ud.a) >= tr.srvs.size() || !tr.d16_srvs.requires_exact(static_cast<unsigned>(ud.a))))
            return;
    }
    if (tr.heapwords && !tr.heap) return log("x2w-vdesc %016llx: `heapwords 1` without `heap 1`: translation refused", tr.key);
    static const bool rt_off = x2w_local::env_on("X2W_DRAW_RT_OFF");
    if (rt_off && (!tr.uavs.empty() || tr.heap || std::any_of(tr.uds.begin(), tr.uds.end(), [](const Translation::Ud& ud) {
            return ud.kind == Translation::buf_word3 || ud.kind == Translation::uav_word3 || ud.kind == Translation::uav_base ||
                   ud.kind == Translation::uav_stride; }))) {
        log("x2w-draw-rt %016llx: not used (X2W_DRAW_RT_OFF)", tr.key);
        return;
    }
    const int mesh_stage = load_mesh_stages(tr, manifest);
    if (mesh_stage < 0 || (!mesh_stage && (!cache_path(path, L"", tr.key, L".vs.cso") || FAILED(D3DReadFileToBlob(path, &tr.vs))))) return;
    if (!mesh_stage && (!cache_path(path, L"", tr.key, L".ps.cso") || FAILED(D3DReadFileToBlob(path, &tr.ps)))) return;
    const HRESULT hr = create_draw_root(tr);
    if (FAILED(hr)) {
        log("x2w-draw root signature for %016llx failed 0x%08lx", tr.key, static_cast<unsigned long>(hr));
        return;
    }
    tr.state = 1;
    if (std::any_of(tr.uds.begin(), tr.uds.end(), [](const Translation::Ud& ud) { return ud.kind == Translation::buf_word3; }))
        InterlockedIncrement(&rt_loaded_word3);
    if (!tr.uavs.empty()) {
        InterlockedIncrement(&rt_loaded_uav);
        log("x2w-draw-rt pipeline %016llx: %zu uavs in root parameter %u", tr.key, tr.uavs.size(), tr.uav_parameter);
    }
    if (tr.heap) {
        InterlockedIncrement(&rt_loaded_heap);
        log("x2w-draw-rt pipeline %016llx: heap 1 in root parameters %u..%u", tr.key, tr.heap_parameter, tr.heap_parameter + 2);
        if (tr.heapwords) log("x2w-vdesc pipeline %016llx: heapwords 1, vdesc in root parameter %u", tr.key, tr.vdesc_parameter);
    }
    if (previously_missing) InterlockedDecrement(&translation_missing);
    InterlockedIncrement(&translation_ready);
    log("x2w-draw pipeline %016llx translated: %zu entry values, %zu resources, %zu static samplers, sampler_heap=%u",
        tr.key, tr.uds.size(), tr.srvs.size(), tr.samplers.statics.size(), tr.samplers.heap ? 1u : 0u);
}

u64 pipeline_key(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes,
                 const std::vector<unsigned char>& root_blob)
{
    // Keep pre-fix desktop bytecode and manifests out of this ABI's cache.
    // tools/startup/gfx2hlsl.py hashes the same bytes, without a trailing NUL.
    constexpr char revision[] = "x2w-startup-shader-3";
    const u64 seed = fnv_bytes(revision, sizeof(revision) - 1);
    return fnv_bytes(root_blob.data(), root_blob.size(), fnv_bytes(b, b_bytes, fnv_bytes(a, a_bytes, seed)));
}

// draw_lock held. Both pipeline forms use the same request writer and shader cache.
Translation* cached_translation(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes,
                                const std::vector<unsigned char>& root_blob, const std::string& layout = {})
{
    u64 key = pipeline_key(a, a_bytes, b, b_bytes, root_blob);
    // A stream's input layout (stream_input_layout) feeds the vertex fetch, so it is part of the key.
    if (!layout.empty()) key = fnv_bytes(layout.data(), layout.size(), key);
    auto& entry = translation_by_key()[key];
    if (!entry) {
        entry = new Translation;
        entry->key = key;
        // A legacy cache hit still needs source blobs for late RTV specialization.
        if (publish_shader_request(key, L".b", b, b_bytes) &&
            publish_shader_request(key, L".root", root_blob.data(), root_blob.size()) &&
            (layout.empty() || publish_shader_request(key, L".ia", layout.data(), layout.size())))
            publish_shader_request(key, L".a", a, a_bytes);
        load_translation(*entry);
    }
    return entry;
}

LONG compute_requests_published, compute_requests_present, compute_requests_failed, compute_blob_pointer_words;

// Device slot 67 (console compute program), 12 64-bit words: 0 (root signature), 1-2 and 3-4 (two state
// blocks and sizes, a size of 0 meaning absent), 5-6 (program code and size), 7-8 (a second blob and size),
// 9 and 10 (pointers whose targets are not decoded) and 11 (a value). The request carries .desc (the 12
// words with the seven pointer words always zeroed, so no process address reaches a cache file or the key),
// .sa, .sb, .data and .root, then .c67 (the code) last as the ready marker, for an offline compute stage
// (tools/startup/gfx2hlsl.py). Any declared part that cannot be read refuses the whole request before a
// file is written. The pipeline stays
// a placeholder. Like the graphics path, a key is published once per cache whether or not a result exists.
constexpr u64 compute_code_limit = 1u << 20, compute_data_limit = 1u << 20, compute_state_limit = 0x1000;

// The key frames every part with its length, so moving bytes from one part to the next changes it.
u64 fnv_part(const std::vector<unsigned char>& part, u64 key)
{
    const u64 bytes = part.size();
    return fnv_bytes(part.data(), part.size(), fnv_bytes(&bytes, sizeof(bytes), key));
}

// A fault-safe copy of title memory: the parts are read once, then hashed, scanned and written from the copy,
// so a title write in between cannot leave a file that does not match its key.
bool copy_title_bytes(void* to, const void* from, std::size_t bytes) noexcept
{
    __try {
        std::memcpy(to, from, bytes);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
    return true;
}

// Aligned qwords of a published blob that name committed memory of this process (one VirtualQuery each, so
// no fault is taken): such a value makes the key run-dependent. They are counted and published, not refused
// (shortcut: hash-like values can name memory by chance; refuse once a decoder names the pointer fields).
LONG pointer_like_qwords(const std::vector<unsigned char>& blob)
{
    LONG found = 0;
    for (std::size_t at = 0; at + 8 <= blob.size(); at += 8) {
        u64 value;
        std::memcpy(&value, blob.data() + at, sizeof(value));
        MEMORY_BASIC_INFORMATION info{};
        if (value >= 0x10000 && value < 0x800000000000ull &&
            VirtualQuery(reinterpret_cast<const void*>(value), &info, sizeof(info)) == sizeof(info) && info.State == MEM_COMMIT)
            ++found;
    }
    return found;
}

SRWLOCK compute_request_lock = SRWLOCK_INIT;
std::unordered_set<u64>& compute_keys_seen() { static std::unordered_set<u64> s; return s; }

void publish_compute_request(const void* desc)
{
    wchar_t marker[MAX_PATH];
    if (!render_mode() || !cache_path(marker, L"req_", 0, L".c67")) return;  // publishing disabled
    static LONG logged;
    const char* refusal = nullptr;
    u64 words[12]{};
    std::vector<unsigned char> root_blob;
    if (!desc || !x2w_local::readable(desc, sizeof(words)) || !copy_title_bytes(words, desc, sizeof(words)))
        refusal = "description unreadable";
    else {
        if (Wrapper* root = as_wrapper(reinterpret_cast<const void*>(words[0]))) {
            AcquireSRWLockShared(&note_lock);
            const auto found = root_blobs().find(root);
            if (found != root_blobs().end()) root_blob = found->second;
            ReleaseSRWLockShared(&note_lock);
        }
        if (root_blob.empty()) refusal = "no root signature blob";
    }
    struct Part { int pointer; u64 limit; bool required; const char *zero, *over, *unreadable; };
    static constexpr Part parts[4] = {
        {5, compute_code_limit, true, "code size 0", "code size over 1 MiB", "code unreadable"},
        {7, compute_data_limit, true, "data size 0", "data size over 1 MiB", "data unreadable"},
        {1, compute_state_limit, false, nullptr, "state block A size over 4 KiB", "state block A unreadable"},
        {3, compute_state_limit, false, nullptr, "state block B size over 4 KiB", "state block B unreadable"},
    };
    std::vector<unsigned char> copies[4];  // code, data, state A, state B
    for (int i = 0; i < 4 && !refusal; ++i) {
        const Part& p = parts[i];
        const u64 bytes = words[p.pointer + 1];
        const auto* at = reinterpret_cast<const void*>(words[p.pointer]);
        if (!bytes) refusal = p.required ? p.zero : nullptr;
        else if (bytes > p.limit) refusal = p.over;
        else {
            copies[i].resize(static_cast<std::size_t>(bytes));
            if (!x2w_local::readable(at, copies[i].size()) || !copy_title_bytes(copies[i].data(), at, copies[i].size()))
                refusal = p.unreadable;
        }
    }
    if (refusal) {
        InterlockedIncrement(&compute_requests_failed);
        if (first_calls(logged, 12)) log("x2w-draw compute request not written: %s", refusal);
        return;
    }
    const auto& [code, data, state_a, state_b] = copies;
    // Pointer words never reach .desc or the key; 9 and 10 are not followed (shortcut: two descriptions that
    // differ only in what they point to share a key; publish those targets once they are decoded).
    std::vector<unsigned char> fields(sizeof(words));
    words[0] = words[1] = words[3] = words[5] = words[7] = words[9] = words[10] = 0;
    std::memcpy(fields.data(), words, sizeof(words));
    constexpr char compute_revision[] = "x2w-startup-shader-3cs";  // tools/startup/gfx2hlsl.py compute_key
    u64 key = fnv_bytes(compute_revision, sizeof(compute_revision) - 1);
    const std::vector<unsigned char>* order[] = {&code, &data, &root_blob, &fields, &state_a, &state_b};
    for (const auto* part : order) key = fnv_part(*part, key);
    // A dedicated lock and an in-memory set: a key seen in this process costs no file access, and draws
    // under draw_lock are not held up by request writes.
    const auto write = [key](const wchar_t* ext, const std::vector<unsigned char>& part) {
        static const unsigned char none = 0;  // a valid buffer for a 0-byte part
        return publish_shader_request(key, ext, part.empty() ? &none : part.data(), part.size());
    };
    AcquireSRWLockExclusive(&compute_request_lock);
    bool present = compute_keys_seen().count(key) != 0;
    bool written = present;
    if (!present) {
        present = cache_path(marker, L"req_", key, L".c67") && GetFileAttributesW(marker) != INVALID_FILE_ATTRIBUTES;
        // Parts left by an interrupted earlier write are kept; the marker comes last, so the request completes here.
        written = present || (write(L".desc", fields) && write(L".sa", state_a) && write(L".sb", state_b) &&
                              write(L".data", data) && write(L".root", root_blob) && write(L".c67", code));
        if (written) compute_keys_seen().insert(key);
    }
    ReleaseSRWLockExclusive(&compute_request_lock);
    const LONG pointers = written && !present ? pointer_like_qwords(data) + pointer_like_qwords(state_a) + pointer_like_qwords(state_b) : 0;
    if (pointers) InterlockedAdd(&compute_blob_pointer_words, pointers);
    InterlockedIncrement(present ? &compute_requests_present : written ? &compute_requests_published : &compute_requests_failed);
    if (first_calls(logged, 12))
        log("x2w-draw compute request %016llx %s (code %llu bytes, data %llu bytes, state blocks %llu + %llu bytes, "
            "pointer-like blob qwords %ld)", key, present ? "already present" : written ? "published" : "not written",
            words[6], words[8], words[2], words[4], pointers);
}

// Slot 57 fixed descriptors consumed by a decoded DispatchMesh signature.
// The vertex span must be absent. A null/zero PS span is valid for depth-only work.
void request_slot57_mesh(Wrapper* pipeline, const std::vector<RootParameter>* bound_root)
{
    wchar_t marker[MAX_PATH];
    if (!bound_root || !cache_path(marker, L"req_", 0, L".m57")) return;
    const PipelineInfo* info = pipeline_info(pipeline);
    for (int depth = 0; info && info->path == 55 && info->base && depth < 8; ++depth)
        info = pipeline_info(info->base);
    constexpr std::size_t fields = 15;  // through the MS length at +0x70
    if (!info || info->path != 57 || info->desc.size() < fields * sizeof(u64)) return;
    u64 words[fields];
    std::memcpy(words, info->desc.data(), sizeof(words));
    if (words[9] || words[10]) return;  // VS at +0x48
    const Wrapper* root = as_wrapper(reinterpret_cast<const void*>(words[0]));
    if (!root || root_info(root) != bound_root) return;
    std::vector<unsigned char> root_blob;
    {
        AcquireSRWLockShared(&note_lock);
        if (const auto found = root_blobs().find(root); found != root_blobs().end()) root_blob = found->second;
        ReleaseSRWLockShared(&note_lock);
    }
    if (root_blob.empty()) return;
    std::vector<unsigned char> mesh, pixel;
    const auto copy_stage = [&words](int field, bool required, std::vector<unsigned char>& out) {
        const auto* pointer = reinterpret_cast<const void*>(words[field]);
        const auto bytes = words[field + 1];
        if (!bytes) return !required && !pointer;
        if (bytes > (1u << 20) || !x2w_local::readable(pointer, bytes)) return false;
        out.resize(static_cast<std::size_t>(bytes));
        return copy_title_bytes(out.data(), pointer, out.size());
    };
    if (!copy_stage(13, true, mesh) || !copy_stage(5, false, pixel)) return;
    constexpr char revision[] = "x2w-startup-mesh-57-1";
    const u64 key = fnv_part(root_blob, fnv_part(pixel, fnv_part(mesh, fnv_bytes(revision, sizeof(revision) - 1))));
    static SRWLOCK request_lock = SRWLOCK_INIT;
    static std::unordered_set<u64> written;
    x2w_startup::ExclusiveNoteLock guard(request_lock);
    if (written.count(key)) return;
    const unsigned char none = 0;
    if (publish_shader_request(key, L".ps57", pixel.empty() ? &none : pixel.data(), pixel.size()) &&
        publish_shader_request(key, L".root", root_blob.data(), root_blob.size()) &&
        publish_shader_request(key, L".m57", mesh.data(), mesh.size())) {
        written.insert(key);
        static LONG logged;
        if (first_calls(logged, 12))
            log("x2w-draw mesh request %016llx: slot57, MS %zu bytes, PS %zu bytes",
                key, mesh.size(), pixel.size());
    }
}

// The INPUT_LAYOUT subobject as the request text tools/startup/gfx2hlsl.py reads: "ia1 <count>", then one
// "<semantic> <index> <format> <slot> <offset> <class> <step>" line per element. False when it cannot be read.
bool stream_input_layout(const unsigned char* subobject, std::string& text)
{
    D3D12_INPUT_LAYOUT_DESC layout;
    std::memcpy(&layout, subobject + 8, sizeof(layout));
    if (!layout.NumElements) return true;
    if (layout.NumElements > D3D12_IA_VERTEX_INPUT_STRUCTURE_ELEMENT_COUNT ||
        !x2w_local::readable(layout.pInputElementDescs, sizeof(D3D12_INPUT_ELEMENT_DESC) * layout.NumElements)) return false;
    text = "ia1 " + std::to_string(layout.NumElements) + "\n";
    std::string shown;
    for (UINT i = 0; i < layout.NumElements; ++i) {
        const D3D12_INPUT_ELEMENT_DESC& e = layout.pInputElementDescs[i];
        std::size_t length = 0;
        while (length < 64 && x2w_local::readable(e.SemanticName + length, 1) && e.SemanticName[length]) {
            const char c = e.SemanticName[length];
            if (!(std::isalpha(static_cast<unsigned char>(c)) || c == '_' || (length && std::isdigit(static_cast<unsigned char>(c))))) return false;
            ++length;
        }
        if (!length || length == 64) return false;
        char line[160];
        std::snprintf(line, sizeof(line), "%.*s %u %u %u %u %u %u\n", static_cast<int>(length), e.SemanticName, e.SemanticIndex,
                      static_cast<unsigned>(e.Format), e.InputSlot, e.AlignedByteOffset, static_cast<unsigned>(e.InputSlotClass),
                      e.InstanceDataStepRate);
        text += line;
        std::snprintf(line, sizeof(line), "%s%u/%u/%u/%u", i ? " " : "", static_cast<unsigned>(e.Format), e.InputSlot,
                      e.AlignedByteOffset, static_cast<unsigned>(e.InputSlotClass));
        shown += line;
    }
    static LONG logged;
    if (first_calls(logged, 24))
        log("x2w-draw stream input layout: %u elements (format/slot/offset/class) %s", layout.NumElements, shown.c_str());
    return true;
}

// RASTERIZER1 stores DepthBias as a float. The desktop PSO takes an int, so only an integral value
// in the int range is representable.
bool integral_depth_bias(float bias, INT& out)
{
    if (!(bias == bias)) return false;
    const double value = static_cast<double>(bias);
    if (value < -2147483648.0 || value > 2147483647.0) return false;
    const INT integral = static_cast<INT>(value);
    if (static_cast<double>(integral) != value) return false;
    out = integral;
    return true;
}

// Resolve the shader pointers and wrapped root while the caller still owns the stream.
Translation* stream_translation(const D3D12_PIPELINE_STATE_STREAM_DESC& desc, D3D12_BLEND_DESC& blend, bool& has_blend,
                                D3D12_DEPTH_STENCIL_DESC& depth_stencil, bool& has_depth_stencil, DXGI_FORMAT& dsv_format,
                                D3D12_RASTERIZER_DESC& rasterizer, bool& has_rasterizer)
{
    // RASTERIZER1 (type 27) is the same width as RASTERIZER. The width comes from the table.
    static_assert(stream_sizes[D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER] == sizeof(unsigned) + sizeof(D3D12_RASTERIZER_DESC));
    const auto* stream = static_cast<const unsigned char*>(desc.pPipelineStateSubobjectStream);
    if (!desc.SizeInBytes || desc.SizeInBytes > 0x2000 || desc.SizeInBytes % 8 ||
        !x2w_local::readable(stream, desc.SizeInBytes)) return nullptr;

    const Wrapper* root = nullptr;
    D3D12_SHADER_BYTECODE stages[2]{};
    std::string layout;
    unsigned seen = 0;
    for (std::size_t at = 0; at < desc.SizeInBytes;) {
        unsigned type;
        std::memcpy(&type, stream + at, sizeof(type));
        // Type 27 is RASTERIZER1. It is outside the table; its width is the RASTERIZER entry.
        const bool rasterizer1 = type == 27;
        if (!rasterizer1 && type >= std::size(stream_sizes)) return nullptr;
        if (type >= 32) return nullptr;
        const std::size_t size = rasterizer1 ? stream_sizes[D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER] : stream_sizes[type];
        if (size > desc.SizeInBytes - at || (seen & (1u << type))) return nullptr;
        seen |= 1u << type;
        if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE) {
            std::memcpy(&root, stream + at + 8, sizeof(root));
        } else if (type >= D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS && type <= D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS) {
            D3D12_SHADER_BYTECODE shader;
            std::memcpy(&shader, stream + at + 8, sizeof(shader));
            if (shader.pShaderBytecode || shader.BytecodeLength) {
                if (type > D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS || shader.BytecodeLength < 32 ||
                    shader.BytecodeLength > (1u << 20) || !x2w_local::readable(shader.pShaderBytecode, shader.BytecodeLength) ||
                    std::memcmp(shader.pShaderBytecode, "DXBC", 4)) return nullptr;
                stages[type - D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS] = shader;
            }
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND) {
            std::memcpy(&blend, stream + at + sizeof(type), sizeof(blend));
            has_blend = true;
        } else if (type == 11 || type == 21) {  // DEPTH_STENCIL (56B) / DEPTH_STENCIL1 (64B)
            std::memcpy(&depth_stencil, stream + at + sizeof(type), sizeof(depth_stencil));
            has_depth_stencil = true;
            if (type == 21) {  // DEPTH_STENCIL1 appends DepthBoundsTestEnable; it has no translated equivalent
                BOOL bounds = FALSE;
                std::memcpy(&bounds, stream + at + sizeof(type) + sizeof(depth_stencil), sizeof(bounds));
                if (bounds) return nullptr;
            }
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT) {
            static const bool upstream_layout = x2w_local::env_on("X2W_UPSTREAM_STREAM_LAYOUT");
            // Unreadable: translate without it; gfx2hlsl refuses a vertex shader that needs it.
            if (upstream_layout && !stream_input_layout(stream + at, layout)) {
                layout.clear();
                log("x2w-draw stream input layout unreadable; requesting the translation without it");
            }
        } else if (type == 16) {  // DEPTH_STENCIL_FORMAT (8B)
            DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
            std::memcpy(&fmt, stream + at + sizeof(type), sizeof(fmt));
            dsv_format = desktop_format(fmt);
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER || type == 27) {
            // Both encodings describe the same rasterizer state.
            if (has_rasterizer) return nullptr;
            // RASTERIZER (48B from stream_sizes) / RASTERIZER1 (same width, float depth bias).
            D3D12_RASTERIZER_DESC parsed{};
            if (sizeof(type) + sizeof(parsed) > size) return nullptr;
            std::memcpy(&parsed, stream + at + sizeof(type), sizeof(parsed));
            if (type == 27) {
                float bias = 0;
                std::memcpy(&bias, &parsed.DepthBias, sizeof(bias));
                INT integral = 0;
                if (!integral_depth_bias(bias, integral)) return nullptr;
                parsed.DepthBias = integral;
            }
            rasterizer = parsed;
            has_rasterizer = true;
        }
        at += size;
    }
    if (!root || !stages[0].pShaderBytecode || !stages[1].pShaderBytecode) return nullptr;
    std::vector<unsigned char> root_blob;
    AcquireSRWLockShared(&note_lock);
    const auto found = root_blobs().find(root);
    if (found != root_blobs().end()) root_blob = found->second;
    ReleaseSRWLockShared(&note_lock);
    if (root_blob.empty()) return nullptr;
    lock_draw(100000 + __LINE__);
    auto* result = cached_translation(stages[0].pShaderBytecode, stages[0].BytecodeLength,
                                      stages[1].pShaderBytecode, stages[1].BytecodeLength, root_blob, layout);
    ReleaseSRWLockExclusive(&draw_lock);
    return result;
}

void decode_draw_state(BoundPipeline& bound, Wrapper* pipeline, const PipelineInfo* info)
{
    using namespace x2w_pipeline_state;
    Words words;
    if (!info || info->path != 57 || !read_words(info->state_a.data(), info->state_a.size(), words)) return;
    bound.has_words = true;
    bound.mesh_layout = words.mesh;
    bound.ds = decode_depth_stencil(words.depth, words.stencil, words.refmask, words.refmask_bf);
    bound.rs = decode_rasterizer(words.mode, words.clip, words.bias_clamp, words.bias_scale, words.bias_offset, draw_state::switches().flip);
    bound.raw_mode = words.mode;
    bound.raw_clip = words.clip;
    bound.masked_blend = decode_blend(words.blend, words.target_mask, words.shader_mask, true, false);
    const D3D12_DEPTH_STENCIL_DESC& d = bound.derived_ds ? bound.patch_ds : bound.ds.desc;
    const bool ds_known = bound.derived_ds || bound.ds.ok;
    draw_state::add(draw_state::decoded);
    if (words.mesh) draw_state::add(draw_state::decoded_mesh);
    if (ds_known)
        draw_state::add(!d.DepthEnable ? draw_state::class_depth_off : d.DepthWriteMask ? draw_state::class_depth_write : draw_state::class_depth_read);
    if (ds_known && d.StencilEnable) draw_state::add(draw_state::class_stencil);
    if (bound.rs.ok && bound.rs.desc.CullMode != D3D12_CULL_MODE_NONE) draw_state::add(draw_state::class_cull);
    if (bound.masked_blend.ok && bound.masked_blend.desc.RenderTarget[0].BlendEnable) draw_state::add(draw_state::class_blend);
    if (bound.masked_blend.masked_targets) draw_state::add(draw_state::class_masked);
    if (bound.masked_blend.dual_source) draw_state::add(draw_state::class_dual);
    if (!bound.ds.ok) draw_state::add(draw_state::refused_depth);
    if (!bound.rs.ok) draw_state::add(draw_state::refused_raster);
    if (!bound.masked_blend.ok) draw_state::add(draw_state::refused_blend);
    if (bound.ds.depth_bounds) draw_state::add(draw_state::depth_bounds_ignored);
    if (bound.ds.masks_differ) draw_state::add(draw_state::stencil_masks_differ);
    if (bound.rs.clip_split) draw_state::add(draw_state::clip_split);
    if (bound.rs.bias_rounded) draw_state::add(draw_state::bias_rounded);
    static LONG logged_state;
    if (first_calls(logged_state, 16)) {
        const auto& r = bound.rs.desc;
        char derived_rs[96] = "";
        if (bound.derived_rs)  // the slot 55 rasterizer part replaces the register state under _CULL
            std::snprintf(derived_rs, sizeof(derived_rs), " (derived part wins: fill %d cull %d ccw %d clip %d)", bound.patch_rs.FillMode,
                          bound.patch_rs.CullMode, bound.patch_rs.FrontCounterClockwise, bound.patch_rs.DepthClipEnable);
        log("x2w-draw-state pipeline %p%s: DB_DEPTH_CONTROL %08x STENCIL %08x REFMASK %08x/%08x -> %s depth %d write %d func %d "
            "stencil %d%s; MODE %08x CLIP %08x bias %g/%g/%g -> %s cull %d ccw %d bias %d/%g/%g clip %d%s; masks %08x & %08x -> "
            "rt0 0x%x%s%s", static_cast<void*>(pipeline), words.mesh ? " (mesh layout)" : "", words.depth, words.stencil, words.refmask,
            words.refmask_bf, bound.ds.ok ? "ok" : bound.ds.reason, d.DepthEnable, d.DepthWriteMask, d.DepthFunc, d.StencilEnable,
            bound.derived_ds ? " (derived part)" : "", words.mode, words.clip, words.bias_clamp, words.bias_scale, words.bias_offset,
            bound.rs.ok ? "ok" : bound.rs.reason, r.CullMode, r.FrontCounterClockwise, r.DepthBias, r.DepthBiasClamp, r.SlopeScaledDepthBias,
            r.DepthClipEnable, derived_rs, words.target_mask, words.shader_mask, bound.masked_blend.desc.RenderTarget[0].RenderTargetWriteMask,
            bound.masked_blend.dual_source ? " dual-source" : "", bound.masked_blend.ok ? "" : " (blend refused)");
    }
}

// draw_lock held.
BoundPipeline* translation_for(Wrapper* pipeline)
{
    auto& by_pipeline = translation_by_pipeline();
    const auto known = by_pipeline.find(pipeline);
    if (known != by_pipeline.end()) return known->second;
    auto* bound = new BoundPipeline;
    Translation*& result = bound->translation;
    const PipelineInfo* info = pipeline_info(pipeline);
    const PipelineInfo* own = info;
    {
        // The pipeline's own description first (a derived pipeline overrides state), then its base's state block.
        std::size_t where = 0;
        const char* from = "none";
        const char* refusal = nullptr;
        unsigned refused_targets = 0;
        for (int depth = 0; info && info->path == 55 && info->base && depth < 8; ++depth) {
            if (!bound->has_blend && x2w_derived_blend::last_blend(info->desc.data(), info->desc.size(), bound->blend, where)) {
                from = "derived blend patch";
                bound->has_blend = true;
            }
            using namespace x2w_pipeline_state;
            const unsigned char* records = info->desc.data();
            const std::size_t record_bytes = info->desc.size();
            if (!bound->derived_ds && last_part(records, record_bytes, part_depth_stencil) != SIZE_MAX) {
                bound->derived_ds = derived_depth_stencil(records, record_bytes, bound->patch_ds);
                draw_state::add(bound->derived_ds ? draw_state::derived_ds : draw_state::derived_invalid);
            }
            if (!bound->derived_rs && last_part(records, record_bytes, part_rasterizer) != SIZE_MAX) {
                bound->derived_rs = derived_rasterizer(records, record_bytes, bound->patch_rs);
                draw_state::add(bound->derived_rs ? draw_state::derived_rs : draw_state::derived_invalid);
            }
            info = pipeline_info(info->base);
        }
        bound->blend_overridden = bound->has_blend || (info && info->has_stream_blend);
        if (!bound->has_blend && info && info->has_stream_blend) {
            from = "stream blend subobject";
            bound->blend = info->stream_blend;
        } else if (!bound->has_blend && info && info->path == 57 &&
                   x2w_startup::state_block_blend(info->state_a.data(), info->state_a.size(), false, &bound->blend, &refusal, &refused_targets)) {
            // The colour-blend register words of a slot 57 pipeline (state_block_blend.h, #1216).
            from = "colour-blend registers";
            where = x2w_startup::state_block_blend_offset;
        } else if (!bound->has_blend && info && find_blend(info->state_a, &bound->blend, &where)) from = "state block A";
        else if (!bound->has_blend && info && find_blend(info->state_b, &bound->blend, &where)) from = "state block B";
        bound->has_blend = from[0] != 'n';
        decode_draw_state(*bound, pipeline, info);
        static LONG logged;
        if (first_calls(logged, 24)) {
            const auto& t = bound->blend.RenderTarget[0];
            log("x2w-draw pipeline %p (creation slot %d, base slot %d): blend from %s at +0x%zx: enable %d src %d dest %d op %d / alpha %d %d %d "
                "mask 0x%x; blocks %zu + %zu bytes", static_cast<void*>(pipeline), own ? own->path : 0, info ? info->path : 0, from, where,
                t.BlendEnable, t.SrcBlend, t.DestBlend, t.BlendOp, t.SrcBlendAlpha, t.DestBlendAlpha, t.BlendOpAlpha, t.RenderTargetWriteMask,
                info ? info->state_a.size() : 0, info ? info->state_b.size() : 0);
            if (refusal || refused_targets)
                log("x2w-draw   colour-blend registers not used: %s; refused targets 0x%x", refusal ? refusal : "refused", refused_targets);
            if (info && !bound->has_blend) {
                char hex[3 * 0x1a0 + 1];
                hex_text(info->state_a.data(), info->state_a.size() < 0x1a0 ? info->state_a.size() : 0x1a0, hex, sizeof(hex));
                log("x2w-draw   state A: %s", hex);
                hex_text(info->state_b.data(), info->state_b.size() < 0x1a0 ? info->state_b.size() : 0x1a0, hex, sizeof(hex));
                log("x2w-draw   state B: %s", hex);
                if (own != info && own) {
                    hex_text(own->desc.data(), own->desc.size() < 0x100 ? own->desc.size() : 0x100, hex, sizeof(hex));
                    log("x2w-draw   derived: %s", hex);
                }
            }
        }
    }
    if (info && info->path == 86) {
        result = info->stream_translation;
        if (info->has_stream_depth_stencil) {
            bound->depth_stencil = info->stream_depth_stencil;
            bound->has_depth_stencil = true;
        }
        if (info->has_stream_rasterizer) {
            bound->rasterizer = info->stream_rasterizer;
            bound->has_rasterizer = true;
        }
        bound->dsv_format = info->stream_dsv_format;
    }
    // X2W_DEPTH_STATE=1 (#1999, item A): the depth-stencil registers of a slot 57 pipeline's state block A. A slot 57
    // pipeline has no desktop DEPTH_STENCIL subobject, so has_depth_stencil can only be set here; the test keeps any
    // desktop state that was found first. A block the decoder refuses leaves the desktop default (depth and stencil off).
    if (info && info->path == 57 && !bound->has_depth_stencil && x2w_local::env_on("X2W_DEPTH_STATE")) {
        const auto chosen = x2w_pipeline_state::select_depth_stencil(true, info->state_a.data(), info->state_a.size());
        static LONG logged_applied, logged_refused;
        if (chosen.applied) {
            bound->depth_stencil = chosen.desc;
            bound->has_depth_stencil = bound->depth_from_registers = true;
            bound->has_stencil_ref = chosen.has_stencil_ref;
            bound->stencil_ref = chosen.stencil_ref;
        }
        if (first_calls(chosen.applied ? logged_applied : logged_refused, 24)) {
            if (chosen.applied)
                log("x2w-draw pipeline %p: depth-stencil from state block A: depth %d func %d write %d, stencil %d reference 0x%x%s%s",
                    static_cast<void*>(pipeline), chosen.desc.DepthEnable, chosen.desc.DepthFunc, chosen.desc.DepthWriteMask,
                    chosen.desc.StencilEnable, chosen.stencil_ref, chosen.depth_bounds ? "; depth bounds not applied" : "",
                    chosen.masks_differ ? "; back-face masks or reference differ, front used" : "");
            else
                log("x2w-draw pipeline %p: depth-stencil registers not used (%s); keeping the desktop state", static_cast<void*>(pipeline),
                    chosen.reason);
        }
    }
    if (info && info->path == 57 && info->desc.size() >= 0x60) {
        u64 words[12];
        std::memcpy(words, info->desc.data(), sizeof(words));
        Wrapper* root = as_wrapper(reinterpret_cast<const void*>(words[0]));
        bound->guest_root = root ? root_info(root) : nullptr;
        std::vector<unsigned char> root_blob;
        if (root) {
            AcquireSRWLockShared(&note_lock);
            const auto found = root_blobs().find(root);
            if (found != root_blobs().end()) root_blob = found->second;
            ReleaseSRWLockShared(&note_lock);
        }
        const auto* a = reinterpret_cast<const void*>(words[5]);
        const auto* b = reinterpret_cast<const void*>(words[9]);
        const std::size_t a_bytes = static_cast<std::size_t>(words[6]), b_bytes = static_cast<std::size_t>(words[10]);
        if (!root_blob.empty() && a_bytes && b_bytes && a_bytes <= (1u << 20) && b_bytes <= (1u << 20) && x2w_local::readable(a, a_bytes) &&
            x2w_local::readable(b, b_bytes)) {
            result = cached_translation(a, a_bytes, b, b_bytes, root_blob);
        }
    }
    by_pipeline[pipeline] = bound;
    return bound;
}

// The request translation_for writes on a direct draw, for a pipeline that slot 61 refuses. Only direct draws poll
// for the manifest afterwards. Count first lookups separately from published request files.
LONG indirect_first_lookups;
void request_translation(Wrapper* pipeline)
{
    AcquireSRWLockExclusive(&draw_lock);
    const bool known = translation_by_pipeline().count(pipeline) != 0;
    translation_for(pipeline);
    ReleaseSRWLockExclusive(&draw_lock);
    if (!known) InterlockedIncrement(&indirect_first_lookups);
    static LONG lookups;
    if (first_calls(lookups, 16))
        log("x2w-render ExecuteIndirect looks up translation for pipeline %p (%s)", static_cast<void*>(pipeline),
            known ? "already known" : "new");
}

// Side-effect-free pipeline key lookup for ExecuteIndirect refusal attribution: never writes
// translation request files and returns 0 when the pipeline has no known console shader key.
u64 known_pipeline_key(Wrapper* pipeline)
{
    if (!pipeline) return 0;
    if (pipeline->own().mesh86_key) return pipeline->own().mesh86_key;
    if (pipeline->own().compute_trace_key) return pipeline->own().compute_trace_key;
    if (pipeline->own().real) return 0;
    AcquireSRWLockShared(&draw_lock);
    const auto& by_pipeline = translation_by_pipeline();
    const auto known = by_pipeline.find(pipeline);
    u64 key = known != by_pipeline.end() && known->second && known->second->translation
        ? known->second->translation->key : 0;
    ReleaseSRWLockShared(&draw_lock);
    if (!key) {
        const PipelineInfo* info = pipeline_info(pipeline);
        if (info && info->path == 86 && info->stream_translation) key = info->stream_translation->key;
    }
    return key;
}

bool needs_rtv_variant(DXGI_FORMAT format)
{
    switch (format) {
    case DXGI_FORMAT_R32G32B32A32_FLOAT: case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R32G32_FLOAT: case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_UINT: case DXGI_FORMAT_R32G32B32A32_SINT:
    case DXGI_FORMAT_R32G32B32_UINT: case DXGI_FORMAT_R32G32B32_SINT:
    case DXGI_FORMAT_R16G16B16A16_UINT: case DXGI_FORMAT_R16G16B16A16_SINT:
    case DXGI_FORMAT_R32G32_UINT: case DXGI_FORMAT_R32G32_SINT:
    case DXGI_FORMAT_R10G10B10A2_UINT:
    case DXGI_FORMAT_R8G8B8A8_UINT: case DXGI_FORMAT_R8G8B8A8_SINT:
    case DXGI_FORMAT_R16G16_UINT: case DXGI_FORMAT_R16G16_SINT:
    case DXGI_FORMAT_R32_UINT: case DXGI_FORMAT_R32_SINT:
    case DXGI_FORMAT_R8G8_UINT: case DXGI_FORMAT_R8G8_SINT:
    case DXGI_FORMAT_R16_UINT: case DXGI_FORMAT_R16_SINT:
    case DXGI_FORMAT_R8_UINT: case DXGI_FORMAT_R8_SINT:
        return true;
    default:
        return false;
    }
}

Translation* translation_for_targets(Translation* base, const Pass& pass)
{
    if (!base) return nullptr;
    const bool no_pixel_output = base->state == 1 && base->ms && !base->ps;
    if (no_pixel_output || std::none_of(pass.target_formats, pass.target_formats + pass.target_count, needs_rtv_variant)) {
        if (manifest_retry_due(*base))
            load_translation(*base);
        return base;
    }
    std::string request = "rtv1 " + std::to_string(base->key);
    for (UINT i = 0; i < pass.target_count; ++i)
        request += " " + std::to_string(static_cast<unsigned>(pass.target_formats[i]));
    const u64 key = fnv_bytes(request.data(), request.size());
    auto& entry = translation_by_key()[key];
    if (!entry) {
        entry = new Translation;
        entry->key = key;
        load_translation(*entry, request);
        if (entry->state != 1) {
            publish_shader_request(key, L".rtv", request.data(), request.size());
            log("x2w-draw no RTV translation for %016llx (%s)", key, request.c_str());
        }
    } else if (manifest_retry_due(*entry)) {
        load_translation(*entry, request);
    }
    return entry;
}

bool apply_draw_state(const BoundPipeline& bound, const Pass& pass, D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc,
                      const x2w_pipeline_state::Switches& s)
{
    using namespace x2w_pipeline_state;
    const DXGI_FORMAT dsv_format = desc.DSVFormat;
    bool changed = false;
    if (s.blend && bound.has_words && !bound.blend_overridden && bound.masked_blend.ok) {
        desc.BlendState = bound.masked_blend.desc;
        if (bound.masked_blend.dual_source) {
            // D3D12 blends two colours only into target 0; the words are the same for every target then.
            desc.BlendState.IndependentBlendEnable = FALSE;
            if (pass.target_count > 1) draw_state::add(draw_state::dual_multi_target);
        }
        changed = true;
    }
    if (s.depth && !bound.has_depth_stencil && (bound.derived_ds || (bound.has_words && bound.ds.ok))) {
        D3D12_DEPTH_STENCIL_DESC d = bound.derived_ds ? bound.patch_ds : bound.ds.desc;
        // A depth target no forwarded slot 51 clear has initialised holds undefined depth: no test against it
        // (unless X2W_DRAW_STATE_DEPTH_UNCLEARED=1); the draw keeps depth and stencil off, as before.
        const bool target = dsv_format != DXGI_FORMAT_UNKNOWN;
        const bool uncleared = target && !s.uncleared && !depth_target_cleared(pass.dsv_resource);
        if (uncleared && (d.DepthEnable || d.StencilEnable)) draw_state::add(draw_state::draws_depth_uncleared);
        const unsigned adjusted = adjust_for_target(d, target && !uncleared, format_has_stencil(dsv_format), pass.dsv_read_only);
        if ((adjusted & adjust_no_target) && !uncleared) draw_state::add(draw_state::draws_depth_no_target);
        if (adjusted & adjust_no_stencil) draw_state::add(draw_state::draws_stencil_no_plane);
        if (adjusted & adjust_read_only) draw_state::add(draw_state::draws_read_only);
        desc.DepthStencilState = d;
        changed = true;
    }
    if (s.cull && (bound.derived_rs || (bound.has_words && bound.rs.ok))) {
        D3D12_RASTERIZER_DESC r = bound.derived_rs ? bound.patch_rs : bound.rs.desc;
        if (bound.derived_rs && s.flip) r.FrontCounterClockwise = !r.FrontCounterClockwise;
        if (r.ConservativeRaster != D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF) {
            r.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;  // tier not checked
            draw_state::add(draw_state::conservative_dropped);
        }
        r.MultisampleEnable = FALSE;  // single-sample targets only (desc.SampleDesc.Count = 1)
        r.ForcedSampleCount = 0;
        // A rectangle list is drawn as two triangles per rectangle; the console rasterizes it as a
        // rectangle, which is taken as never culled (assumed).
        if (pass.topology == 0x11 && r.CullMode != D3D12_CULL_MODE_NONE) {
            r.CullMode = D3D12_CULL_MODE_NONE;
            draw_state::add(draw_state::draws_rect_cull_off);
        }
        desc.RasterizerState = r;
        changed = true;
    } else if (s.depthclip && (bound.derived_rs || (bound.has_words && bound.rs.ok))) {
        const BOOL clip = (bound.derived_rs ? bound.patch_rs : bound.rs.desc).DepthClipEnable ? TRUE : FALSE;
        if (desc.RasterizerState.DepthClipEnable != clip) {
            desc.RasterizerState.DepthClipEnable = clip;
            changed = true;
        }
    }
    return changed;
}

HRESULT draw_state_fallback(u64 translation, HRESULT first, const D3D12_GRAPHICS_PIPELINE_STATE_DESC& decoded,
                            const D3D12_GRAPHICS_PIPELINE_STATE_DESC& fallback, ID3D12PipelineState*& pipeline)
{
    if (SUCCEEDED(first)) {
        draw_state::add(draw_state::pso_new);
        return first;
    }
    pipeline = nullptr;
    const HRESULT hr = device_of(the_device)->CreateGraphicsPipelineState(&fallback, IID_PPV_ARGS(&pipeline));
    draw_state::add(SUCCEEDED(hr) ? draw_state::pso_failed_fallback : draw_state::pso_failed_both);
    static LONG logged_failure;
    if (first_calls(logged_failure, 16)) {
        const auto& b = decoded.BlendState.RenderTarget[0];
        const auto& d = decoded.DepthStencilState;
        const auto& r = decoded.RasterizerState;
        log("x2w-draw-state decoded state refused for %016llx (0x%08lx): blend rt0 %d/%d/%d mask 0x%x independent %d, depth %d write %d "
            "func %d stencil %d, cull %d ccw %d bias %d/%g/%g clip %d; old state -> 0x%08lx", translation, static_cast<unsigned long>(first),
            b.BlendEnable, b.SrcBlend, b.DestBlend, b.RenderTargetWriteMask, decoded.BlendState.IndependentBlendEnable, d.DepthEnable,
            d.DepthWriteMask, d.DepthFunc, d.StencilEnable, r.CullMode, r.FrontCounterClockwise, r.DepthBias, r.DepthBiasClamp,
            r.SlopeScaledDepthBias, r.DepthClipEnable, static_cast<unsigned long>(hr));
    }
    return hr;
}

struct PsoBuild {
    BoundPipeline* bound = nullptr;  // never freed (translation_by_pipeline keeps them)
    u64 key = 0;                     // pipeline_for's key in bound->pipelines
    u64 translation = 0;             // for the log
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};  // points at the translation's blobs and root signature, which are never freed
    bool has_fallback = false;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC fallback{};
};
struct PsoSlot {
    int state = 0;  // 1 compiling, 2 done, 3 queued for an X2W_DRAW_ASYNC worker
    ID3D12PipelineState* pipeline = nullptr;
};
SRWLOCK pso_build_lock = SRWLOCK_INIT;  // inside draw_lock when both are held
CONDITION_VARIABLE pso_built = CONDITION_VARIABLE_INIT;
// Slots are never erased: a thread that missed before another installed the pipeline finds it done.
std::map<std::pair<const BoundPipeline*, u64>, PsoSlot>& pso_slots()
{
    static std::map<std::pair<const BoundPipeline*, u64>, PsoSlot> m;
    return m;
}

void test_hold(const char* what, u64 key)
{
    wchar_t hold[128];
    const DWORD n = GetEnvironmentVariableW(L"X2W_CX_TEST_HOLD_EVENT", hold, 128);
    if (!n || n >= 128) return;
    if (const HANDLE event = OpenEventW(SYNCHRONIZE, FALSE, hold)) {
        const DWORD waited = WaitForSingleObject(event, 20000);
        CloseHandle(event);
        log("%s %016llx: load held by X2W_CX_TEST_HOLD_EVENT (%s)", what, key, waited == WAIT_OBJECT_0 ? "released" : "timed out");
    }
}

bool draw_async()
{
    static const bool on = !x2w_local::env_on("X2W_DRAW_PSO_LOCKED") &&
                           (x2w_local::env_on("X2W_DRAW_ASYNC") || (x2w_local::env_on("X2W_CX_ASYNC") && !x2w_local::env_on("X2W_DRAW_ASYNC_OFF")));
    return on;
}
struct DrawAsync {
    LONG skipped = 0, since_report = 0;
};
std::unordered_map<u64, DrawAsync>& draw_async_keys()  // pso_build_lock; by translation key
{
    static std::unordered_map<u64, DrawAsync> m;
    return m;
}
// The depth format a translated draw uses: the pass's own view, else the one the pipeline was created with.
inline DXGI_FORMAT pass_dsv_format(const BoundPipeline& bound, const Pass& pass)
{
    return pass.dsv_format != DXGI_FORMAT_UNKNOWN ? pass.dsv_format : bound.dsv_format;
}

std::deque<PsoBuild>& pso_queue()  // pso_build_lock
{
    static std::deque<PsoBuild> q;
    return q;
}
CONDITION_VARIABLE pso_wanted = CONDITION_VARIABLE_INIT;

void compile_pso(const PsoBuild& build, PsoSlot& slot);
DWORD WINAPI pso_thread(void*)
{
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    for (;;) {
        AcquireSRWLockExclusive(&pso_build_lock);
        while (pso_queue().empty()) SleepConditionVariableSRW(&pso_wanted, &pso_build_lock, INFINITE, 0);
        const PsoBuild build = pso_queue().front();
        pso_queue().pop_front();
        PsoSlot& slot = pso_slots()[{build.bound, build.key}];
        const bool mine = slot.state == 3;  // else another thread claimed it
        if (mine) slot.state = 1;
        ReleaseSRWLockExclusive(&pso_build_lock);
        if (mine) compile_pso(build, slot);
    }
}
// draw_translated without draw_lock: queues `build` unless it is queued, compiling or built; counts the skip.
void queue_pso_build(const PsoBuild& build)
{
    static const bool started = [] {
        for (int i = 0; i < 2; ++i)
            if (const HANDLE t = CreateThread(nullptr, 0, pso_thread, nullptr, 0, nullptr)) CloseHandle(t);
        log("x2w-draw X2W_DRAW_ASYNC: 2 pipeline build threads; a draw whose desktop pipeline is not built yet is skipped");
        return true;
    }();
    (void)started;
    AcquireSRWLockExclusive(&pso_build_lock);
    PsoSlot& slot = pso_slots()[{build.bound, build.key}];
    const bool queue = slot.state == 0;
    if (queue) {
        slot.state = 3;
        pso_queue().push_back(build);
    }
    DrawAsync& a = draw_async_keys()[build.translation];
    const bool first = !a.skipped;
    ++a.skipped, ++a.since_report;
    ReleaseSRWLockExclusive(&pso_build_lock);
    if (queue) WakeConditionVariable(&pso_wanted);
    static LONG logged;
    if (first && first_calls(logged, 32))
        log("x2w-draw %016llx: desktop pipeline not built yet, draw skipped and the build queued (X2W_DRAW_ASYNC)", build.translation);
}
// Heartbeat: draws skipped while their pipeline compiled, since the last report, per translation key.
void report_draw_async()
{
    if (!draw_async()) return;
    std::vector<std::pair<u64, LONG>> keys;
    LONG total = 0, all_time = 0;
    AcquireSRWLockExclusive(&pso_build_lock);
    for (auto& entry : draw_async_keys()) {
        all_time += entry.second.skipped;
        if (!entry.second.since_report) continue;
        keys.push_back({entry.first, entry.second.since_report});
        total += entry.second.since_report;
        entry.second.since_report = 0;
    }
    const std::size_t waiting = pso_queue().size();
    ReleaseSRWLockExclusive(&pso_build_lock);
    if (!total) return;
    std::sort(keys.begin(), keys.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    char line[1400];
    int n = std::snprintf(line, sizeof(line), "x2w-draw draws skipped while compiling since the last report (X2W_DRAW_ASYNC): %ld over %zu keys "
                          "(%ld this run; %zu builds waiting):", total, keys.size(), all_time, waiting);
    for (const auto& k : keys) {
        if (n > static_cast<int>(sizeof(line)) - 40) {
            n += std::snprintf(line + n, sizeof(line) - n, " ...");
            break;
        }
        n += std::snprintf(line + n, sizeof(line) - n, " %016llx=%ld", k.first, k.second);
    }
    log("%s", line);
}

void build_pipeline(const PsoBuild& build)
{
    AcquireSRWLockExclusive(&pso_build_lock);
    PsoSlot& slot = pso_slots()[{build.bound, build.key}];  // std::map references stay valid
    while (slot.state == 1) SleepConditionVariableSRW(&pso_built, &pso_build_lock, INFINITE, 0);
    if (slot.state == 2) {
        ReleaseSRWLockExclusive(&pso_build_lock);
        return;
    }
    slot.state = 1;  // from 0, or 3 (queued for a worker: claimed here)
    ReleaseSRWLockExclusive(&pso_build_lock);
    compile_pso(build, slot);
}
void compile_pso(const PsoBuild& build, PsoSlot& slot)
{
    test_hold("x2w-draw", build.translation);
    LARGE_INTEGER start, end, frequency;
    QueryPerformanceCounter(&start);
    ID3D12PipelineState* pipeline = nullptr;
    HRESULT hr = device_of(the_device)->CreateGraphicsPipelineState(&build.desc, IID_PPV_ARGS(&pipeline));
    if (build.has_fallback) hr = draw_state_fallback(build.translation, hr, build.desc, build.fallback, pipeline);
    QueryPerformanceCounter(&end);
    QueryPerformanceFrequency(&frequency);
    log("x2w-draw desktop pipeline for %016llx, %u targets (format %d) -> 0x%08lx (dsv %d, depth %d%s), compiled in %lld ms outside draw_lock",
        build.translation, build.desc.NumRenderTargets, build.desc.NumRenderTargets ? build.desc.RTVFormats[0] : 0,
        static_cast<unsigned long>(hr), static_cast<int>(build.desc.DSVFormat), build.desc.DepthStencilState.DepthEnable ? 1 : 0,
        build.has_fallback ? ", decoded state" : "", static_cast<long long>((end.QuadPart - start.QuadPart) * 1000 / frequency.QuadPart));
    check_device("translated CreateGraphicsPipelineState", 0);
    AcquireSRWLockExclusive(&pso_build_lock);
    slot.pipeline = pipeline;
    slot.state = 2;
    LONG skipped = 0;
    if (draw_async())
        if (const auto found = draw_async_keys().find(build.translation); found != draw_async_keys().end()) skipped = found->second.skipped;
    ReleaseSRWLockExclusive(&pso_build_lock);
    WakeAllConditionVariable(&pso_built);
    if (skipped) log("x2w-draw %016llx ready (X2W_DRAW_ASYNC); %ld draws skipped while compiling", build.translation, skipped);
}

bool pipeline_desc(const BoundPipeline& bound, Translation& tr, const Pass& pass, DXGI_FORMAT dsv_format,
                   D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc, bool indexed_rect = false);

// draw_lock held. With `deferred`, a pipeline not built yet is described there and nullptr is returned.
// indexed_rect (upstream #2259): the draw is an indexed rectangle list, expanded by a geometry stage.
ID3D12PipelineState* pipeline_for(BoundPipeline& bound, Translation& tr, const Pass& pass, PsoBuild* deferred = nullptr,
                                  bool indexed_rect = false)
{
    if (tr.ms) return mesh_pso_for(tr);
    const DXGI_FORMAT dsv_format = pass_dsv_format(bound, pass);
    u64 key = pass.target_count;
    for (UINT i = 0; i < pass.target_count; ++i) key = key * 131 + static_cast<u64>(pass.target_formats[i]);
    key = fnv_bytes(&dsv_format, sizeof(dsv_format), key);
    // A state decoded from registers needs a depth view to act on; with none bound the pass keeps the desktop default.
    const bool use_depth_stencil = bound.has_depth_stencil && (!bound.depth_from_registers || dsv_format != DXGI_FORMAT_UNKNOWN);
    if (use_depth_stencil) key = fnv_bytes(&bound.depth_stencil, sizeof(bound.depth_stencil), key);
    if (bound.has_rasterizer) key = fnv_bytes(&bound.rasterizer, sizeof(bound.rasterizer), key);
    key = fnv_bytes(&indexed_rect, sizeof(indexed_rect), key);
    D3D12_GRAPHICS_PIPELINE_STATE_DESC decoded{};
    bool changed = false;
    const auto& state_switches = draw_state::switches();
    if (state_switches.depth || state_switches.cull || state_switches.blend || state_switches.depthclip) {
        if (!pipeline_desc(bound, tr, pass, dsv_format, decoded, indexed_rect)) {
            bound.pipelines[key] = nullptr;
            return nullptr;
        }
        changed = apply_draw_state(bound, pass, decoded, state_switches);
        if (changed) {
            key = fnv_bytes(&decoded.BlendState, sizeof(decoded.BlendState), key);
            key = fnv_bytes(&decoded.DepthStencilState, sizeof(decoded.DepthStencilState), key);
            key = fnv_bytes(&decoded.RasterizerState, sizeof(decoded.RasterizerState), key);
        }
    }
    const auto found = bound.pipelines.find(key);
    if (found != bound.pipelines.end()) return found->second;
    {
        AcquireSRWLockShared(&pso_build_lock);
        const auto built = pso_slots().find({&bound, key});
        const bool done = built != pso_slots().end() && built->second.state == 2;
        ID3D12PipelineState* const ready = done ? built->second.pipeline : nullptr;
        ReleaseSRWLockShared(&pso_build_lock);
        if (done) {
            bound.pipelines[key] = ready;
            return ready;
        }
    }
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    if (!pipeline_desc(bound, tr, pass, dsv_format, desc, indexed_rect)) {
        bound.pipelines[key] = nullptr;
        return nullptr;
    }
    if (deferred) {
        deferred->bound = &bound;
        deferred->key = key;
        deferred->translation = tr.key;
        deferred->desc = changed ? decoded : desc;
        deferred->has_fallback = changed;
        if (changed) deferred->fallback = desc;
        return nullptr;
    }
    ID3D12PipelineState* pipeline = nullptr;
    HRESULT hr = device_of(the_device)->CreateGraphicsPipelineState(changed ? &decoded : &desc, IID_PPV_ARGS(&pipeline));
    if (changed) hr = draw_state_fallback(tr.key, hr, decoded, desc, pipeline);
    log("x2w-draw desktop pipeline for %016llx, %u targets (format %d) -> 0x%08lx (dsv %d, depth %d%s)", tr.key,
        pass.target_count, pass.target_count ? pass.target_formats[0] : 0, static_cast<unsigned long>(hr),
        static_cast<int>(dsv_format), (changed ? decoded : desc).DepthStencilState.DepthEnable ? 1 : 0, changed ? ", decoded state" : "");
    check_device("translated CreateGraphicsPipelineState", 0);
    bound.pipelines[key] = pipeline;
    return pipeline;
}

// The desktop PSO description of a translated (vertex) draw with the state of before: blend found in the
// state block, depth-stencil from a stream PSO, no culling.
bool pipeline_desc(const BoundPipeline& bound, Translation& tr, const Pass& pass, DXGI_FORMAT dsv_format,
                   D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc, bool indexed_rect)
{
    desc = {};
    desc.pRootSignature = tr.root;
    desc.VS = {tr.vs->GetBufferPointer(), tr.vs->GetBufferSize()};
    desc.PS = {tr.ps->GetBufferPointer(), tr.ps->GetBufferSize()};
    if (indexed_rect) {
        if (!tr.indexed_rect_geometry_attempted) {
            tr.indexed_rect_geometry_attempted = true;
            x2w_startup::ComPtr<ID3DBlob> errors;
            const HRESULT hr = x2w_startup::compile_indexed_rect_geometry(tr.vs, tr.ps, &tr.indexed_rect_geometry, &errors);
            if (FAILED(hr))
                log("x2w-draw %016llx: indexed rectangle geometry unavailable (0x%08lx) %s", tr.key,
                    static_cast<unsigned long>(hr), errors ? static_cast<const char*>(errors->GetBufferPointer()) : "");
        }
        if (!tr.indexed_rect_geometry) return false;
        desc.GS = {tr.indexed_rect_geometry->GetBufferPointer(), tr.indexed_rect_geometry->GetBufferSize()};
    }
    // Blend state as found in the pipeline's state block, else opaque. Depth-stencil state and
    // DSVFormat come from the stream PSO (or bound DSV), else disabled. Raster state comes from
    // the stream RASTERIZER / RASTERIZER1 subobject, else solid fill, no culling, depth clip on.
    for (auto& target : desc.BlendState.RenderTarget)
        target = {FALSE, FALSE, D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD, D3D12_BLEND_ONE, D3D12_BLEND_ZERO,
                  D3D12_BLEND_OP_ADD, D3D12_LOGIC_OP_NOOP, D3D12_COLOR_WRITE_ENABLE_ALL};
    if (bound.has_blend) desc.BlendState = bound.blend;
    static const bool default_blend = x2w_local::env_on("X2W_DRAW_DEFAULT_BLEND");
    if (!bound.has_blend && default_blend)
        for (auto& target : desc.BlendState.RenderTarget)
            target = {TRUE, FALSE, D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_OP_ADD, D3D12_BLEND_ONE,
                      D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_OP_ADD, D3D12_LOGIC_OP_NOOP, D3D12_COLOR_WRITE_ENABLE_ALL};
    // Channels the translated exports leave unwritten stay out of the write mask (#1674).
    if (!x2w_startup::apply_export_write_masks(desc.BlendState, tr.write_masks, pass.target_count)) {
        log("x2w-draw export write masks incompatible with shared logic op for %016llx", tr.key);
        return false;
    }
    // A state decoded from registers needs a depth view to act on; with none bound the pass keeps the desktop default.
    const bool use_depth_stencil = bound.has_depth_stencil && (!bound.depth_from_registers || dsv_format != DXGI_FORMAT_UNKNOWN);
    if (use_depth_stencil) desc.DepthStencilState = bound.depth_stencil;
    desc.SampleMask = ~0u;
    if (bound.has_rasterizer) {
        desc.RasterizerState = bound.rasterizer;
    } else {
        desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        desc.RasterizerState.DepthClipEnable = TRUE;
    }
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = pass.target_count;
    for (UINT i = 0; i < pass.target_count; ++i) desc.RTVFormats[i] = pass.target_formats[i];
    desc.DSVFormat = dsv_format;
    desc.SampleDesc.Count = 1;
    return true;
}

// Private shader-visible heap for the translated draws' tables, used as a ring.
constexpr UINT draw_heap_size = 1u << 18;
ID3D12DescriptorHeap* draw_heap;
Wrapper* draw_heap_device;
UINT draw_heap_increment;
volatile LONG draw_heap_cursor;

bool ensure_draw_heap(Wrapper* device)
{
    // The translated pipeline and upload ring are scoped to the current
    // startup render device. A list from another device cannot use this heap.
    if (!device || device != the_device) return false;
    if (draw_heap) return device == draw_heap_device;
    const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, draw_heap_size + 8 + 8 * 90000, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    if (FAILED(device_of(the_device)->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&draw_heap)))) return false;
    draw_heap_device = device;
    draw_heap_increment = device_of(the_device)->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return true;
}

// A recorded table is immutable. Keep it alive until its submission completes,
// including when the title resets the command list with a different allocator.
using DrawSamplerHeap = x2w_startup::ComPtr<ID3D12DescriptorHeap>;
std::unordered_map<ID3D12CommandList*, std::vector<DrawSamplerHeap>> recorded_draw_sampler_heaps;
struct SubmittedDrawSamplerHeaps {
    x2w_startup::ComPtr<ID3D12Fence> fence;
    std::vector<DrawSamplerHeap> heaps;
};
std::vector<SubmittedDrawSamplerHeaps> submitted_draw_sampler_heaps;

void discard_draw_sampler_heaps(ID3D12CommandList* list)
{
    lock_draw(100000 + __LINE__);
    recorded_draw_sampler_heaps.erase(list);
    ReleaseSRWLockExclusive(&draw_lock);
}

// draw_lock held. The lists have already been executed on this queue. A closed list can be
// submitted again until its Reset, so each submit takes its own references and the recording
// keeps its own until discard_draw_sampler_heaps.
void retire_draw_sampler_heaps(ID3D12CommandQueue* queue, const std::vector<ID3D12CommandList*>& lists)
{
    submitted_draw_sampler_heaps.erase(std::remove_if(submitted_draw_sampler_heaps.begin(),
        submitted_draw_sampler_heaps.end(), [](const SubmittedDrawSamplerHeaps& batch) {
            return batch.fence && batch.fence->GetCompletedValue() != 0;
        }), submitted_draw_sampler_heaps.end());
    SubmittedDrawSamplerHeaps batch;
    for (auto* list : lists) {
        auto it = recorded_draw_sampler_heaps.find(list);
        if (it == recorded_draw_sampler_heaps.end()) continue;
        for (const auto& heap : it->second) batch.heaps.push_back(heap);
    }
    if (batch.heaps.empty()) return;
    x2w_startup::ComPtr<ID3D12Device> device;
    HRESULT hr = queue->GetDevice(IID_PPV_ARGS(&device));
    if (SUCCEEDED(hr)) hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&batch.fence));
    if (SUCCEEDED(hr)) hr = queue->Signal(batch.fence.Get(), 1);
    if (FAILED(hr)) {
        // Without a completion point the submitted descriptors still belong
        // to the queue. Retain them instead of reclaiming them early.
        batch.fence.Reset();
        log("x2w-draw sampler heap retirement fence failed 0x%08lx", static_cast<unsigned long>(hr));
    }
    submitted_draw_sampler_heaps.push_back(std::move(batch));
}

struct TextureUpload {
    u64 hash = 0, generation = ~0ull;
    LONG checked_frame = -1000;
    bool done = false, tiled = false;
    ID3D12CommandList* pending_list = nullptr;
    u64 revision = 0;
    bool staged = false;       // the last upload attempt put the title memory into the desktop texture
    bool gpu_written = false;  // a list copy or resolve wrote the texture; its title memory is stale and must not be uploaded
};
std::unordered_map<ID3D12Resource*, TextureUpload>& texture_uploads() { static std::unordered_map<ID3D12Resource*, TextureUpload> m; return m; }
u64 address_upload_revision;

// A texture published in a newly supported console layout is not filled through the shared upload ring.
// The ring is reused without waiting for any queue, and a queue held behind a fence can execute a
// recorded copy long after later uploads have passed over its source bytes. Each such upload owns
// its upload buffer instead. The buffer and the texture it fills stay referenced by the recording,
// and by every submit of that recording until the queue it went to has completed the submit, as
// for the sampler heaps above.
struct AddressUpload {
    x2w_startup::ComPtr<ID3D12Resource> buffer, texture;
    u64 revision;
    bool pinned = false;
};
std::unordered_map<ID3D12CommandList*, std::vector<AddressUpload>> recorded_address_uploads;
struct SubmittedAddressUploads {
    x2w_startup::ComPtr<ID3D12Fence> fence;
    std::vector<AddressUpload> uploads;
};
std::vector<SubmittedAddressUploads> submitted_address_uploads;

// X2W_TEST_ADDRESS_UPLOAD_RING=1 (fixtures only): fill by-address textures from the shared upload ring,
// without the buffers above. The control of startup_address_console_gate runs with it.
bool address_uploads_from_ring()
{
    static const bool on = GetEnvironmentVariableW(L"X2W_TEST_ADDRESS_UPLOAD_RING", nullptr, 0) != 0;
    return on;
}

void discard_address_uploads(ID3D12CommandList* list)
{
    AcquireSRWLockExclusive(&draw_lock);
    const auto recorded = recorded_address_uploads.find(list);
    if (recorded != recorded_address_uploads.end()) {
        for (const auto& upload : recorded->second) {
            const auto state = texture_uploads().find(upload.texture.Get());
            if (state != texture_uploads().end() && state->second.pending_list == list) {
                state->second.done = false;
                state->second.pending_list = nullptr;
            }
        }
        recorded_address_uploads.erase(recorded);
    }
    ReleaseSRWLockExclusive(&draw_lock);
}

// draw_lock held. The lists have already been executed on this queue. Each submit takes its own
// references; they are dropped by a later call, once the fence signalled behind the submit has
// completed. A submit whose fence could not be made or signalled keeps its references.
void retire_address_uploads(ID3D12CommandQueue* queue, const std::vector<ID3D12CommandList*>& lists)
{
    static std::size_t peak, released;
    const std::size_t before = submitted_address_uploads.size();
    submitted_address_uploads.erase(std::remove_if(submitted_address_uploads.begin(),
        submitted_address_uploads.end(), [](const SubmittedAddressUploads& batch) {
            return batch.fence && batch.fence->GetCompletedValue() != 0;
        }), submitted_address_uploads.end());
    if (const std::size_t gone = before - submitted_address_uploads.size()) {
        released += gone;
        static LONG reports;
        if (first_calls(reports, 12))
            log("x2w-draw by-address uploads: %zu submits completed and released (%zu so far, %zu still pending, at most %zu pending)",
                gone, released, submitted_address_uploads.size(), peak);
    }
    for (auto* list : lists) {
        const auto it = recorded_address_uploads.find(list);
        if (it == recorded_address_uploads.end()) continue;
        for (const auto& upload : it->second) {
            const auto state = texture_uploads().find(upload.texture.Get());
            if (state == texture_uploads().end()) continue;
            // A recording that writes an older revision may be submitted again. When the
            // last write in a batch differs from the cache, make the next draw restage it.
            auto& cached = state->second;
            cached.done = cached.revision == upload.revision;
            if (cached.pending_list == list) cached.pending_list = nullptr;
        }
    }
    try {
        SubmittedAddressUploads batch;
        for (auto* list : lists) {
            const auto it = recorded_address_uploads.find(list);
            if (it == recorded_address_uploads.end()) continue;
            for (const auto& upload : it->second)
                if (!upload.pinned) batch.uploads.push_back(upload);
        }
        if (batch.uploads.empty()) return;
        x2w_startup::ComPtr<ID3D12Device> device;
        HRESULT hr = queue->GetDevice(IID_PPV_ARGS(&device));
        if (SUCCEEDED(hr)) hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&batch.fence));
        if (SUCCEEDED(hr)) hr = queue->Signal(batch.fence.Get(), 1);
        if (FAILED(hr)) {
            batch.fence.Reset();
            log("x2w-draw by-address upload retirement fence failed 0x%08lx", static_cast<unsigned long>(hr));
        }
        submitted_address_uploads.push_back(std::move(batch));
        peak = (std::max)(peak, submitted_address_uploads.size());
    } catch (const std::bad_alloc&) {
        // Execution already happened. Pin each affected resource once if bookkeeping fails.
        for (auto* list : lists) {
            const auto it = recorded_address_uploads.find(list);
            if (it == recorded_address_uploads.end()) continue;
            for (auto& upload : it->second) {
                if (upload.pinned) continue;
                upload.buffer->AddRef();
                upload.texture->AddRef();
                upload.pinned = true;
            }
        }
        log("x2w-draw by-address upload retirement storage exhausted; uploads pinned");
    }
}

bool resolve_draw_sampler(const x2w_startup::SamplerWords& words, D3D12_SAMPLER_DESC& sampler)
{
    if (words[0] != view_token_magic) {
        sampler = x2w_startup::dynamic_sampler(words);
        return true;
    }
    AcquireSRWLockShared(&note_lock);
    const auto& records = view_records();
    const bool valid = words[2] == 22 && words[1] < records.size() && records[words[1]].slot == 22 &&
                       records[words[1]].desc_bytes == sizeof(sampler);
    if (valid) std::memcpy(&sampler, records[words[1]].desc, sizeof(sampler));
    ReleaseSRWLockShared(&note_lock);
    return valid;
}

// draw_lock held.
bool prepare_draw_sampler_heap(Pass& pass, ID3D12GraphicsCommandList* list, bool reserved_fallback)
{
    auto snapshot = x2w_startup::snapshot_samplers(pass.title_samplers, reserved_fallback);
    if (pass.sampler_snapshot && *pass.sampler_snapshot == snapshot) return true;
    x2w_startup::NullSamplerCounts counts;
    DrawSamplerHeap heap;
    const HRESULT hr = x2w_startup::create_sampler_snapshot_heap(device_of(the_device), snapshot,
        resolve_draw_sampler, heap.GetAddressOf(), counts);
    if (FAILED(hr)) {
        log("x2w-draw sampler heap creation failed 0x%08lx", static_cast<unsigned long>(hr));
        return false;
    }
    static LONG reports;
    if (first_calls(reports, 12))
        log("x2w-draw sampler heap: null_sampler_unbound=%u null_sampler_out_of_range=%u "
            "null_sampler_unreadable=%u null_sampler_invalid_token=%u",
            counts.unbound, counts.out_of_range, counts.unreadable, counts.invalid_token);
    recorded_draw_sampler_heaps[list].push_back(heap);
    pass.sampler_heap = std::move(heap);
    pass.sampler_snapshot = std::make_unique<x2w_startup::SamplerSnapshot>(std::move(snapshot));
    return true;
}

// Copies of title memory made for the current frame, found by (address, bytes). A title may rewrite the same
// address between two draws of one frame (a dynamic constant buffer, a reused vertex or index range), so an
// entry keeps the bytes it was made from and is reused only while the title's memory still matches them. The
// upload ring is write-combined memory that is slow to read, so the comparison is against this copy, not the ring.
struct FrameCopy {
    u64 address = 0;
    u64 offset = 0;
    ID3D12Resource* buffer = nullptr;
    RingTag tag;
    std::shared_ptr<const std::vector<unsigned char>> content;
};
std::unordered_map<u64, FrameCopy>& frame_copies() { static std::unordered_map<u64, FrameCopy> m; return m; }
LONG frame_copies_frame = -1;

enum { srvc_compared, srvc_copied, srvc_in_place, srvc_kinds };
volatile LONG64 srvc_bytes[4][srvc_kinds], srvc_calls[4][srvc_kinds];
// Over the threshold but copied as before: [class][0] no single desktop window (desktop_range), [1] not 16-aligned.
volatile LONG64 srvc_in_place_missed[4][2];
thread_local int frame_copy_class;
void srvc_note(int cls, int kind, std::size_t bytes)
{
    InterlockedAdd64(&srvc_bytes[cls][kind], static_cast<LONG64>(bytes));
    InterlockedIncrement64(&srvc_calls[cls][kind]);
}
// X2W_DRAW_SRV_IN_PLACE=<min bytes> (off by default; 0 or unset = off).
std::size_t srv_in_place_min()
{
    static const std::size_t min = [] {
        char value[24]{};
        const DWORD n = GetEnvironmentVariableA("X2W_DRAW_SRV_IN_PLACE", value, sizeof(value));
        return n && n < sizeof(value) ? static_cast<std::size_t>(std::strtoull(value, nullptr, 0)) : std::size_t{0};
    }();
    return min;
}
// Positive byte threshold for comparing direct-draw snapshots outside draw_lock (opt-in).
std::size_t draw_copy_split_min()
{
    static const std::size_t min = [] {
        char mode[16]{};
        GetEnvironmentVariableA("X2W_UPLOAD_RING", mode, sizeof(mode));
        if (!std::strcmp(mode, "legacy") || ring_env("X2W_UPLOAD_RING_TEST_NO_REPIN", 0)) return std::size_t{0};
        char value[24]{};
        const DWORD n = GetEnvironmentVariableA("X2W_DRAW_COPY_SPLIT_MIN", value, sizeof(value));
        return n && n < sizeof(value) ? static_cast<std::size_t>(std::strtoull(value, nullptr, 0)) : std::size_t{0};
    }();
    return min;
}
bool srv_in_place_all()
{
    static const bool all = x2w_local::env_on("X2W_DRAW_SRV_IN_PLACE_ALL");
    return all;
}
void report_srv_copies()  // the 5 s report (d3d12_xs_device.cpp)
{
    static const bool on = srv_in_place_min() || x2w_local::env_on("X2W_DRAW_SRV_STATS");
    if (!on) return;
    char text[960];
    int used = 0;
    static const char* const names[4] = {"other", "draw", "mesh", "table V#"};
    for (int c = 0; c < 4; ++c) {
        LONG64 b[srvc_kinds], n[srvc_kinds];
        for (int k = 0; k < srvc_kinds; ++k) b[k] = InterlockedExchange64(&srvc_bytes[c][k], 0), n[k] = InterlockedExchange64(&srvc_calls[c][k], 0);
        const LONG64 no_window = InterlockedExchange64(&srvc_in_place_missed[c][0], 0), unaligned = InterlockedExchange64(&srvc_in_place_missed[c][1], 0);
        used += std::snprintf(text + used, sizeof(text) - used,
                              "%s%s: compared %lld MiB/%lld, copied %lld MiB/%lld, in place %lld MiB/%lld, in-place misses no window %lld unaligned %lld",
                              c ? "; " : "", names[c], b[srvc_compared] >> 20, n[srvc_compared], b[srvc_copied] >> 20, n[srvc_copied],
                              b[srvc_in_place] >> 20, n[srvc_in_place], no_window, unaligned);
        if (used >= static_cast<int>(sizeof(text))) used = static_cast<int>(sizeof(text)) - 1;
    }
    log("x2w-draw srv copies since the last report (X2W_DRAW_SRV_IN_PLACE=%zu%s): %s", srv_in_place_min(),
        srv_in_place_all() ? ", _ALL: plain draws too" : ", mesh calls only", text);
}

bool compare_draw_copy(u64 address, std::size_t bytes, u64* offset, ID3D12Resource** buffer)
{
    if (frame_copies_frame != frame_number) return false;
    const u64 key = address * 1099511628211ull ^ bytes;
    FrameCopy seen;
    {
        const auto found = frame_copies().find(key);
        if (found == frame_copies().end() || found->second.address != address || !found->second.content ||
            found->second.content->size() != bytes) return false;
        seen = found->second;
    }
    const LONG frame = frame_number;
    ReleaseSRWLockExclusive(&draw_lock);
    const bool equal = std::memcmp(seen.content->data(), reinterpret_cast<const void*>(address), bytes) == 0;
    lock_draw(100000 + __LINE__);
    if (!equal || frame != frame_number) return false;
    const auto current = frame_copies().find(key);
    if (current == frame_copies().end() || current->second.content != seen.content || !ring_touch(seen.tag)) return false;
    srvc_note(frame_copy_class, srvc_compared, bytes);
    *offset = seen.offset;
    *buffer = seen.buffer;
    return true;
}
// Callers have just probed that [address, address + bytes) is readable (readable_window), and draw_lock is held.
bool frame_copy(u64 address, std::size_t bytes, u64* offset, ID3D12Resource** buffer)
{
    if (frame_copies_frame != frame_number) {
        frame_copies().clear();
        frame_copies_frame = frame_number;
    }
    const u64 key = address * 1099511628211ull ^ bytes;
    const auto* title = reinterpret_cast<const unsigned char*>(address);
    const auto found = frame_copies().find(key);
    const bool known = found != frame_copies().end() && found->second.address == address && found->second.content &&
                       found->second.content->size() == bytes;
    if (known && std::memcmp(found->second.content->data(), title, bytes) == 0 && ring_touch(found->second.tag)) {
        srvc_note(frame_copy_class, srvc_compared, bytes);
        *offset = found->second.offset;
        *buffer = found->second.buffer;
        return true;
    }
    u64 fresh = 0;
    ID3D12Resource* chunk = nullptr;
    RingTag tag;
    auto* ring = reserve_upload_ring(bytes, &fresh, &chunk, &tag);
    if (!ring) return false;
    srvc_note(frame_copy_class, srvc_copied, bytes);
    // Cache insertion and snapshot allocation are optional. Fill the ring directly if they fail.
    const unsigned char* source = title;
    try {
        auto content = std::make_shared<const std::vector<unsigned char>>(title, title + bytes);
        // Use the stored snapshot to fill the ring so the GPU bytes and subsequent comparisons agree.
        auto entry = frame_copies().insert_or_assign(key, FrameCopy{address, fresh, chunk, tag, std::move(content)});
        source = entry.first->second.content->data();
    } catch (const std::bad_alloc&) {
        // The ring slice is already reserved. The draw can proceed without another CPU allocation.
    }
    std::memcpy(ring, source, bytes);
    x2w_memcut::note_copy(frame_number, bytes);
    if (known) {
        static LONG refreshed;
        if (first_calls(refreshed, 4))
            log("x2w-draw frame copy of 0x%llx (%zu bytes) refreshed: the title changed it since an earlier draw of this frame", address, bytes);
    }
    *offset = fresh;
    *buffer = chunk;
    return true;
}
bool frame_copy_unlocked(u64 address, std::size_t bytes, u64* offset, ID3D12Resource** buffer, int lookup_line, int commit_line)
{
    const u64 key = address * 1099511628211ull ^ bytes;
    const auto* title = reinterpret_cast<const unsigned char*>(address);
    std::shared_ptr<const std::vector<unsigned char>> seen;
    u64 seen_offset = 0;
    ID3D12Resource* seen_buffer = nullptr;
    RingTag seen_tag;
    lock_draw(lookup_line);
    if (frame_copies_frame != frame_number) {
        frame_copies().clear();
        frame_copies_frame = frame_number;
    }
    const LONG frame = frame_copies_frame;
    if (const auto found = frame_copies().find(key); found != frame_copies().end() && found->second.address == address &&
                                                     found->second.content && found->second.content->size() == bytes) {
        seen = found->second.content;
        seen_offset = found->second.offset;
        seen_buffer = found->second.buffer;
        seen_tag = found->second.tag;
    }
    ReleaseSRWLockExclusive(&draw_lock);
    if (seen && std::memcmp(seen->data(), title, bytes) == 0 && ring_touch(seen_tag)) {
        srvc_note(frame_copy_class, srvc_compared, bytes);
        *offset = seen_offset;
        *buffer = seen_buffer;
        return true;
    }
    u64 fresh = 0;
    ID3D12Resource* chunk = nullptr;
    RingTag tag;
    auto* ring = reserve_upload_ring(bytes, &fresh, &chunk, &tag);
    if (!ring) return false;
    srvc_note(frame_copy_class, srvc_copied, bytes);
    std::shared_ptr<const std::vector<unsigned char>> content;
    try {
        content = std::make_shared<const std::vector<unsigned char>>(title, title + bytes);
    } catch (const std::bad_alloc&) {
        // The ring slice is already reserved: fill it from the title memory and keep no entry.
    }
    std::memcpy(ring, content ? content->data() : title, bytes);
    x2w_memcut::note_copy(frame_number, bytes);
    if (content) {
        lock_draw(commit_line);
        if (frame_copies_frame == frame) {
            try {
                frame_copies().insert_or_assign(key, FrameCopy{address, fresh, chunk, tag, std::move(content)});
            } catch (const std::bad_alloc&) {
            }
        }
        ReleaseSRWLockExclusive(&draw_lock);
    }
    if (seen) {
        static LONG refreshed;
        if (first_calls(refreshed, 4))
            log("x2w-draw frame copy of 0x%llx (%zu bytes) refreshed: the title changed it since an earlier draw of this frame", address, bytes);
    }
    *offset = fresh;
    *buffer = chunk;
    return true;
}

void buffer_view(D3D12_CPU_DESCRIPTOR_HANDLE dest, ID3D12Resource* buffer, u64 offset, std::size_t bytes)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.Format = DXGI_FORMAT_R32_TYPELESS;
    desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    desc.Buffer.FirstElement = buffer ? offset / 4 : 0;
    desc.Buffer.NumElements = buffer ? static_cast<UINT>((bytes + 3) / 4) : 1;
    desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    device_of(the_device)->CreateShaderResourceView(buffer, &desc, dest);
}

// Texture contents. Linear backing uses desktop copyable footprints; an announced BC7 4 KiB
// surface must be detiled into the footprint first. A texture is uploaded on its first bind and
// again when its memory changed: XG copies announce themselves (x2w_d3d12_texture_written), and
// CPU-backed textures up to 32 MiB are also re-hashed for writes the title makes on its own.
// Recheck single-mip R8 and A8 planes on their first bind each frame and other textures every 15 frames.
// Targets and UAVs are GPU-written and never uploaded.
LONG textures_uploaded, texture_bytes_uploaded, textures_unreadable;
u64 hash_memory(const unsigned char* data, std::size_t bytes)
{
    u64 h = 0xcbf29ce484222325ull;
    const std::size_t words = bytes / 8;
    const u64* w = reinterpret_cast<const u64*>(data);
    for (std::size_t i = 0; i < words; ++i) h = (h ^ w[i]) * 0x100000001b3ull;
    for (std::size_t i = words * 8; i < bytes; ++i) h = (h ^ data[i]) * 0x100000001b3ull;
    return h;
}
x2w_startup::gfx10_local::Layout detile_layout(const D3D12_RESOURCE_DESC& r);  // d3d12_memory_srv.inl (X2W_MEMORY_SRV_DETILE)
bool list_wrote(Wrapper* list, ID3D12Resource* resource);  // list_copy.inl
bool upload_texture(Wrapper* self, Wrapper* resource, ID3D12Resource* texture)
{
    auto* const memory = static_cast<const unsigned char*>(resource->own().title_memory);
    if (InterlockedCompareExchange(&resource->own().refs, 0, 0) <= 0) x2w_memcut::note_after_release(true);
    auto* list = recording(self);
    const D3D12_RESOURCE_DESC desc = texture->GetDesc();
    const auto* original = resource_info(resource);
    const bool tiled_address = resource->own().tiled_address_srv;
    const bool console = tiled_address && original && console_layout_srv(original->console);
    const bool dedicated = console && !address_uploads_from_ring();
    // An address-only single-element texture (2D or 3D) uploads its one texel through the footprint path.
    const bool single = resource->own().tiled_address_srv && x2w_startup::single_element_bytes(desc) != 0;
    const bool linear_srv = resource->own().linear_memory_srv;
    const bool volume = linear_srv && desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    const bool uploadable = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D || single ||
        (desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D && desc.Height == 1 && (desc.DepthOrArraySize == 1 || linear_srv));
    {
        static LONG reasons;
        if ((!memory || (desc.Flags & 0x7) || !uploadable) && !texture_uploads().count(texture) &&
            first_calls(reasons, 12)) {
            texture_uploads()[texture].done = true;
            log("x2w-draw texture not uploaded %p: %llux%u format %d flags 0x%x dim %d title memory %p", static_cast<void*>(texture),
                desc.Width, desc.Height, desc.Format, desc.Flags, desc.Dimension, static_cast<const void*>(memory));
        }
    }
    if (!memory || !list || !uploadable) return !console;
    if (list_wrote(self, texture)) return !console;  // this recording's own copy; not yet submitted
    if (const auto written_by_gpu = texture_uploads().find(texture); written_by_gpu != texture_uploads().end() && written_by_gpu->second.gpu_written) return !console;
    if (desc.Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) return !console;
    const UINT mips = desc.MipLevels ? desc.MipLevels : 1;
    // For Texture3D, depth slices are carried in Footprint.Depth and spanned by SliceFootprints::bytes().
    const UINT slices = (desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D || !desc.DepthOrArraySize) ? 1 : desc.DepthOrArraySize;
    if (mips > 15) return !console;
    const x2w_startup::SliceFootprints plan(device_of(the_device), desc, mips);
    const UINT64 slice_total = plan.size;
    if (slice_total == ~0ull || !slice_total) return !console;
    const u64 stride = (slice_total + 0xFFFF) & ~u64{0xFFFF};
    TextureUpload* cached;
    try {
        cached = &texture_uploads()[texture];
    } catch (const std::bad_alloc&) {
        if (!console) throw;
        return false;
    }
    TextureUpload& state = *cached;
    TextureWrite written;
    AcquireSRWLockShared(&written_lock);
    if (const auto found = written_textures().find(reinterpret_cast<u64>(memory)); found != written_textures().end())
        written = found->second;
    ReleaseSRWLockShared(&written_lock);
    const bool tiled_chain = original && x2w_startup::bc64::supports(original->console);
    // Any other texture slot 70 publishes holds its texels in the console layout its description names
    // (gfx10_layout.h, address_texture.h).
    const bool tiled_bc7 = !tiled_chain && !tiled_address && written.bc7_width != 0;
    const bool tiled = tiled_bc7 || tiled_address || tiled_chain;
    const bool detiled = resource->own().detiled_memory_srv && original;
    const x2w_startup::gfx10_local::Layout console_layout_local = detiled ? detile_layout(original->console) : x2w_startup::gfx10_local::Layout(~0u, false, 0, 0, 0, 0, 0);
    if (resource->own().detiled_memory_srv && !console_layout_local.valid) return !console;
    std::optional<x2w_startup::gfx10::Layout> console_layout;
    if (console) console_layout.emplace(original->console);
    const auto generation = written.generation;
    const x2w_startup::Bc16Surface surface(tiled_bc7 ? written.bc7_width : static_cast<UINT>(desc.Width),
                                         tiled_bc7 ? written.bc7_height : desc.Height);
    if (tiled_bc7) {
        if (!original || !x2w_startup::supports_bc7_4k_s(original->console) ||
            original->console.Width != written.bc7_width || original->console.Height != written.bc7_height ||
            desc.Format != DXGI_FORMAT_BC7_UNORM || desc.Width != surface.columns * surface.block_texels ||
            desc.Height != surface.rows * surface.block_texels) return true;
    }
    if (tiled && !tiled_chain && !single && !console && (plan.count != 1 || plan.rows[0] != surface.rows ||
                  plan.row_bytes[0] != u64{surface.columns} * surface.block_bytes)) return true;
    if (console) {
        // The desktop footprints must hold exactly the elements the console layout addresses.
        if (console_layout->mips != mips || console_layout->slices != slices || plan.count != mips) return false;
        for (UINT i = 0; i < mips; ++i)
            if (!plan.bytes(i) || plan.rows[i] != console_layout->rows(i) ||
                plan.row_bytes[i] != u64{console_layout->columns(i)} * console_layout->bytes_per_element) return false;
    }
    // A texture given the 4 KiB allocation answer is read in the compact plan XG describes and writes for it
    // (small_texture_4k.h); its rows must be the footprint's rows, level by level. Address-only textures
    // (single-element, or a console layout named in Layout) are tiled and keep their own paths.
    x2w_startup::SmallTexturePlan compact;
    const bool is_compact = original && !tiled && x2w_startup::small_texture_4k_plan(original->console, compact);
    if (is_compact) {
        bool rows_match = compact.levels == plan.count;
        for (UINT i = 0; rows_match && i < plan.count; ++i)
            rows_match = compact.level[i].rows == plan.rows[i] && compact.level[i].row_bytes == plan.row_bytes[i];
        if (!rows_match) return true;
    }
    const u64 total = detiled ? console_layout_local.size
        : tiled_chain ? x2w_startup::bc64::Chain(original->console).bytes() : console ? console_layout->bytes()
        : tiled && !single ? surface.size() : is_compact ? compact.bytes : stride * (slices - 1) + slice_total;
    const bool announced = state.generation != generation;
    const UINT elapsed = static_cast<UINT>(frame_number) - static_cast<UINT>(state.checked_frame);
    // A single texel (at most 16 bytes) is rehashed on every bind, so a write between two draws is seen.
    const bool recheck = single || ((tiled_address || tiled_chain) ? elapsed != 0
        : x2w_startup::recheck_cpu_texture(desc, total, elapsed));
    const bool reusable = state.done && (!state.pending_list || state.pending_list == list);
    if (reusable && !announced && !recheck) return true;
    if (readable_window(reinterpret_cast<u64>(memory), static_cast<std::size_t>(total)) < total) {
        if (!state.done) InterlockedIncrement(&textures_unreadable);
        if (console) return false;
        state.done = true;
        state.staged = false;
        state.generation = generation;
        return true;
    }
    if (!console) state.checked_frame = frame_number;
    const u64 hash = hash_memory(memory, static_cast<std::size_t>(total));
    if (reusable && hash == state.hash && state.tiled == tiled) {
        state.checked_frame = frame_number;
        state.generation = generation;
        return true;
    }
    constexpr u64 placement = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
    x2w_upload::Page owned;
    if (dedicated) {
        u64 bytes = 0;
        for (UINT i = 0; i < mips; ++i) bytes += (plan.bytes(i) + placement - 1) & ~(placement - 1);
        bytes *= slices;
        if (FAILED(x2w_upload::make_page(device_of(the_device), bytes, owned))) return false;
        // Store the owning reference before issuing any D3D12 commands that name the buffer.
        try {
            recorded_address_uploads[list].push_back({owned.resource, texture, ++address_upload_revision});
        } catch (const std::bad_alloc&) {
            return false;
        }
        std::memset(owned.cpu, 0, static_cast<std::size_t>(bytes));
    }
    state.staged = false;
    require_state(self, texture, D3D12_RESOURCE_STATE_COPY_DEST);
    if (detiled) {
        // Each element of each level, array slice or volume depth from its console address into the desktop
        // footprint. Element extents are the desktop ones (rows and row bytes of the footprint), which addrlib's
        // ceiling level extents contain.
        const UINT eb = 1u << console_layout_local.elem_log2;
        for (UINT s = 0; s < slices; ++s)
            for (UINT i = 0; i < mips; ++i) {
                const auto& fp = plan.placed[i];
                const u64 bytes = plan.bytes(i);
                u64 offset = 0;
                ID3D12Resource* chunk = nullptr;
                // One reservation per level and slice (at most one ring chunk, 64 MiB; a larger level is refused
                // by the ring, and the texture is retried later).
                auto* rows = reserve_upload_ring(static_cast<std::size_t>(bytes), &offset, &chunk, nullptr);
                if (!rows) {
                    state.done = false;
                    return !console;
                }
                const UINT columns = static_cast<UINT>(plan.row_bytes[i] / eb), depth = fp.Footprint.Depth ? fp.Footprint.Depth : 1;
                for (UINT z = 0; z < depth; ++z)
                    for (UINT y = 0; y < plan.rows[i]; ++y) {
                        auto* row = rows + (u64{z} * plan.rows[i] + y) * fp.Footprint.RowPitch;
                        for (UINT x = 0; x < columns; ++x)
                            std::memcpy(row + u64{x} * eb, memory + console_layout_local.address(i, x, y, volume ? z : s), eb);
                    }
                plan.upload(list, texture, chunk, mips, slices, s, i, offset);
            }
    } else if (console) {
        for (UINT slice = 0; slice < slices; ++slice) {
            for (UINT i = 0; i < mips; ++i) {
                const u64 bytes = plan.bytes(i);
                u64 offset = owned.cursor;
                ID3D12Resource* chunk = nullptr;
                unsigned char* rows = owned.cpu ? owned.cpu + offset : nullptr;
                if (owned.cpu) {
                    owned.cursor += (bytes + placement - 1) & ~(placement - 1);
                } else {
                    rows = reserve_upload_ring(static_cast<std::size_t>(bytes), &offset, &chunk, nullptr);
                    if (!rows) {
                        state.done = false;
                        return false;
                    }
                    std::memset(rows, 0, static_cast<std::size_t>(bytes));
                }
                console_layout->detile(i, slice, memory, rows, plan.placed[i].Footprint.RowPitch);
                plan.upload(list, texture, owned.cpu ? owned.resource.Get() : chunk, mips, slices, slice, i, offset);
            }
        }
        if (dedicated) {
            state.pending_list = list;
            state.revision = address_upload_revision;
        }
    } else if (tiled_chain) {
        const x2w_startup::bc64::Chain chain(original->console);
        for (UINT slice = 0; slice < slices; ++slice) {
            for (UINT i = 0; i < mips; ++i) {
                const u64 bytes = plan.bytes(i);
                u64 offset = 0;
                ID3D12Resource* chunk = nullptr;
                auto* rows = reserve_upload_ring(static_cast<std::size_t>(bytes), &offset, &chunk, nullptr);
                if (!rows) {
                    state.done = false;
                    return true;
                }
                std::memset(rows, 0, static_cast<std::size_t>(bytes));
                chain.copy<false>(i, slice, rows, memory, plan.placed[i].Footprint.RowPitch);
                plan.upload(list, texture, chunk, mips, slices, slice, i, offset);
            }
        }
    } else if (tiled && !single) {
        u64 offset = 0;
        ID3D12Resource* chunk = nullptr;
        const auto bytes = plan.bytes(0);
        auto* rows = reserve_upload_ring(static_cast<std::size_t>(bytes), &offset, &chunk, nullptr);
        if (!rows) {
            state.done = false;
            return true;
        }
        std::memset(rows, 0, static_cast<std::size_t>(bytes));
        surface.detile(memory, rows, plan.placed[0].Footprint.RowPitch);
        plan.upload(list, texture, chunk, 1, 1, 0, 0, offset);
    } else {
        for (UINT s = 0; s < slices; ++s)
            for (UINT i = 0; i < plan.count; ++i) {
                const auto& fp = plan.placed[i];
                const u64 bytes = plan.bytes(i);
                u64 offset = 0;
                ID3D12Resource* chunk = nullptr;
                if (is_compact) {
                    auto* const rows = reserve_upload_ring(static_cast<std::size_t>(bytes), &offset, &chunk, nullptr);
                    if (!rows) {
                        state.done = false;
                        return true;
                    }
                    const auto& level = compact.level[i];
                    for (u64 y = 0; y < level.rows; ++y)
                        std::memcpy(rows + y * fp.Footprint.RowPitch, memory + level.offset + y * level.row_bytes,
                                    static_cast<std::size_t>(level.row_bytes));
                } else if (!ring_copy(reinterpret_cast<u64>(memory) + stride * s + fp.Offset, static_cast<std::size_t>(bytes), &offset, &chunk)) {
                    state.done = false;
                    return true;
                }
                plan.upload(list, texture, chunk, mips, slices, s, i, offset);
            }
    }
    state.hash = hash;
    state.generation = generation;
    state.checked_frame = frame_number;
    state.tiled = tiled;
    state.done = true;
    state.staged = true;
    InterlockedIncrement(&textures_uploaded);
    InterlockedExchangeAdd(&texture_bytes_uploaded, static_cast<LONG>(total >> 10));
    static LONG shown;
    if (first_calls(shown, 12))
        log("x2w-draw texture upload %p %llux%u x%u mips %u format %d from %p (%llu KiB)%s%s", static_cast<void*>(texture), desc.Width,
            desc.Height, slices, mips, desc.Format, static_cast<const void*>(memory), total >> 10, announced ? " (announced write)" : "",
            is_compact ? " (compact plan of the 4 KiB allocation answer)" : "");
    return true;
}

bool heap_texture_upload_on()
{
    static const bool on = [] {
        const bool v = x2w_local::env_on("X2W_HEAP_TEXTURE_UPLOAD");
        if (v) log("x2w-draw X2W_HEAP_TEXTURE_UPLOAD=1: textures referenced by heap views are uploaded (budgeted, once a frame)");
        return v;
    }();
    return on;
}
std::vector<Wrapper*> heap_textures;
std::unordered_set<Wrapper*> heap_texture_set;
std::size_t heap_texture_cursor;
LONG heap_texture_frame = -1;
LONG heap_textures_uploaded;
void note_heap_texture(const ViewRecord& record)
{
    if (!heap_texture_upload_on() || record.slot != 18) return;
    Wrapper* resource = as_wrapper(reinterpret_cast<const void*>(record.resource));
    if (!resource || resource->own().kind != resource_kind || !resource->own().real || !resource->own().title_memory) return;
    const D3D12_RESOURCE_DESC desc = static_cast<ID3D12Resource*>(resource->own().real)->GetDesc();
    if (desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER || (desc.Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
        D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))) return;
    if (heap_texture_set.insert(resource).second) heap_textures.push_back(resource);
}
void upload_heap_textures(Wrapper* self)
{
    if (!heap_texture_upload_on() || heap_textures.empty() || heap_texture_frame == frame_number || !recording(self)) return;
    heap_texture_frame = frame_number;
    static const LONG budget_kib = [] {
        char value[16]{};
        const LONG v = GetEnvironmentVariableA("X2W_HEAP_TEXTURE_BUDGET_KIB", value, sizeof value) ? std::atol(value) : 0L;
        return v > 0 ? v : 65536L;
    }();
    // Time as well as bytes: upload_texture also rehashes unchanged CPU-backed textures on its recheck schedule
    // (up to 32 MiB each), which the byte budget does not see. X2W_HEAP_TEXTURE_BUDGET_US (default 2000) bounds the
    // whole visit under draw_lock; the first texture is always visited.
    static const LONG budget_us = [] {
        char value[16]{};
        const LONG v = GetEnvironmentVariableA("X2W_HEAP_TEXTURE_BUDGET_US", value, sizeof value) ? std::atol(value) : 0L;
        return v > 0 ? v : 2000L;
    }();
    LARGE_INTEGER frequency{}, begin{}, now{};
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&begin);
    const LONG start_kib = texture_bytes_uploaded;
    UINT uploaded = 0;
    std::size_t visited = 0;
    for (; visited < heap_textures.size() && visited < 512 && texture_bytes_uploaded - start_kib < budget_kib; ++visited) {
        if (visited) {
            QueryPerformanceCounter(&now);
            if ((now.QuadPart - begin.QuadPart) * 1000000 / frequency.QuadPart >= budget_us) break;
        }
        Wrapper* resource = heap_textures[heap_texture_cursor++ % heap_textures.size()];
        // Revalidated each visit (wrappers are never freed today; this keeps it safe if that changes).
        if (as_wrapper(resource) != resource || !resource->own().real) continue;
        if (InterlockedCompareExchange(&resource->own().refs, 0, 0) <= 0) continue;  // released: left alone
        auto* texture = static_cast<ID3D12Resource*>(resource->own().real);
        const LONG before = textures_uploaded;
        upload_texture(self, resource, texture);
        if (textures_uploaded == before) continue;
        require_state(self, texture, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        ++uploaded;
    }
    if (!uploaded) return;
    const LONG n = InterlockedExchangeAdd(&heap_textures_uploaded, static_cast<LONG>(uploaded)) + static_cast<LONG>(uploaded);
    static LONG shown;
    if (first_calls(shown, 8))
        log("x2w-draw heap textures: frame %ld uploaded %u (%ld KiB), visited %zu of %zu listed; %ld uploaded so far", frame_number,
            uploaded, texture_bytes_uploaded - start_kib, visited, heap_textures.size(), n);
}

// Per-draw dumps (X2W_DRAW_DUMP_FRAME=<frame>, X2W_DRAW_DUMP_DIR=<folder>): during that frame, render
// target 0 is copied to a readback buffer after each translated draw (at most 48).
// Poll the fence of the queue that actually executed the recording before writing its image.
using x2w_startup::DrawDump;
LONG quiet_dump_frame;
LONG draw_dump_frame()
{
    static const LONG frame = [] {
        char value[16]{};
        return GetEnvironmentVariableA("X2W_DRAW_DUMP_FRAME", value, sizeof(value)) ? std::atol(value) : 0L;
    }();
    return frame ? frame : quiet_dump_frame;
}
void draw_dump_after(Wrapper* self, ID3D12GraphicsCommandList* list, u64 key, UINT count)
{
    Pass& pass = pass_of(self);
    const LONG wanted = draw_dump_frame();
    if (!wanted || frame_number < wanted || frame_number > wanted + 1 || !pass.target0 || draw_dumps().full()) return;
    const D3D12_RESOURCE_DESC desc = pass.target0->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1 ||
        !x2w_draw_bmp::supported(desc.Format)) return;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0, total = 0;
    device_of(the_device)->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &row_bytes, &total);
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = total;
    buffer.Height = 1;
    buffer.DepthOrArraySize = 1;
    buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* readback = nullptr;
    if (FAILED(device_of(the_device)->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                              IID_PPV_ARGS(&readback)))) return;
    x2w_budget::committed(device_of(the_device), readback, D3D12_HEAP_TYPE_READBACK);
    DrawDump dump{};
    dump.buffer.Attach(readback);
    dump.footprint = footprint;
    dump.rows = rows;
    dump.key = key;
    dump.count = count;
    dump.format = desc.Format;
    if (!draw_dumps().record(list, std::move(dump))) return;
    require_state(self, pass.target0, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION target{readback, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    target.PlacedFootprint = footprint;
    D3D12_TEXTURE_COPY_LOCATION source{pass.target0, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    source.SubresourceIndex = 0;
    list->CopyTextureRegion(&target, 0, 0, 0, &source, nullptr);
    require_state(self, pass.target0, D3D12_RESOURCE_STATE_RENDER_TARGET);
}
void draw_dump_flush()
{
    if (!draw_dump_frame()) return;
    auto completed = draw_dumps().take_completed();
    if (completed.empty()) return;
    char folder[MAX_PATH] = ".";
    GetEnvironmentVariableA("X2W_DRAW_DUMP_DIR", folder, sizeof(folder));
    int written = 0;
    for (const DrawDump& dump : completed) {
        const int index = static_cast<int>(dump.index);
        void* mapped = nullptr;
        if (FAILED(dump.buffer->Map(0, nullptr, &mapped))) continue;
        char path[MAX_PATH + 64];
        std::snprintf(path, sizeof(path), "%s\\draw_%02d_%016llx_%u.bmp", folder, index, dump.key, dump.count);
        if (FILE* file = std::fopen(path, "wb")) {
            const auto& fp = dump.footprint.Footprint;
            const auto* pixels = static_cast<const unsigned char*>(mapped) + dump.footprint.Offset;
            const bool complete = x2w_draw_bmp::write(file, pixels, fp.Width, dump.rows, fp.RowPitch, dump.format);
            if (std::fclose(file) == 0 && complete) ++written;
            else std::remove(path);
        }
        const D3D12_RANGE no_writes{};
        dump.buffer->Unmap(0, &no_writes);
    }
    log("x2w-draw per-draw dump: %d of %zu completed target copies written to %s", written, completed.size(), folder);
}

D3D12_GPU_VIRTUAL_ADDRESS desktop_va(u64 address);
bool desktop_range(u64 address, u64 bytes, ID3D12Resource** buffer, u64* offset);
void fill_views(Wrapper* self);
bool title_view_heap(const Pass& pass, u64* address, u64* bytes);
extern ID3D12Resource* vinfo_buffer;
extern UINT views_filled;
extern ID3D12Resource* vdesc_buffer;
bool ensure_vdesc();

struct TitleHeapView {
    D3D12_GPU_VIRTUAL_ADDRESS va = 0;  // the base's desktop address (log only)
    ID3D12Resource* window = nullptr;  // the view: window + offset, bytes
    u64 offset = 0, bytes = 0, heap_bytes = 0;
};
LONG rt_heap_clamped;  // views shortened to the end of the base's window (counted once per frame and heap)
const char* bounded_title_heap(u64 address, u64 bytes, TitleHeapView& out);
const char* draw_title_heap(const Pass& pass, TitleHeapView& out)
{
    u64 address = 0, bytes = 0;
    if (pass.source_view_heap) {
        AcquireSRWLockShared(&note_lock);
        for (const HeapInfo& h : heap_infos())
            if (h.wrapper == pass.source_view_heap && h.title_cpu) {
                address = h.title_cpu;
                bytes = u64{h.logical_count} * h.increment;  // the title's count (count may be the driver-clamped one)
            }
        ReleaseSRWLockShared(&note_lock);
    }
    if (!address && !title_view_heap(pass, &address, &bytes)) return "title view heap not reachable";
    return bounded_title_heap(address, bytes, out);
}

// The bounded view over the title heap [address, address + bytes), for draws and translated dispatches
// (d3d12_compute.inl). Cached per frame and heap; draw_lock held.
const char* bounded_title_heap(u64 address, u64 bytes, TitleHeapView& out)
{
    static u64 last_address, last_bytes;
    static TitleHeapView last;
    static const char* last_why;
    static LONG last_frame = -1;
    if (address != last_address || bytes != last_bytes || last_frame != frame_number) {
        last_address = address;
        last_bytes = bytes;
        last_frame = frame_number;
        last = {};
        last.heap_bytes = bytes;
        last.va = desktop_va(address);
        last_why = nullptr;
        ID3D12Resource* window = nullptr;
        u64 offset = 0, view_bytes = bytes;
        if (!last.va) {
            last_why = "title view heap not reachable";
        } else if (!bytes) {
            last_why = "title view heap size unknown";
        } else if (!desktop_range(address, bytes, &window, &offset)) {
            // The heap runs past the window that holds its base: the view ends at that window's end.
            window = nullptr;
            if (desktop_range(address, 16, &window, &offset)) {
                const u64 width = window->GetDesc().Width;
                view_bytes = width > offset ? width - offset : 0;
                InterlockedIncrement(&rt_heap_clamped);
                static LONG shown;
                if (first_calls(shown, 4))
                    log("x2w-draw-rt title heap 0x%llx (%llu bytes) runs past its window: view clamped to %llu bytes", address, bytes, view_bytes);
            } else {
                window = nullptr;
            }
        }
        if (!last_why && (!window || view_bytes < 4)) last_why = "title view heap not in a desktop window";
        else if (!last_why && (offset & 15)) last_why = "title view heap base not 16-byte aligned (bounded raw view)";
        if (!last_why) {
            last.window = window;
            last.offset = offset;
            last.bytes = view_bytes & ~u64{3};
        }
    }
    out = last;
    return last_why;
}

void bounded_raw_srv(ID3D12Resource* buffer, u64 offset, u64 bytes, D3D12_CPU_DESCRIPTOR_HANDLE dest)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC d{};
    d.Format = DXGI_FORMAT_R32_TYPELESS;
    d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    d.Buffer.FirstElement = offset / 4;
    d.Buffer.NumElements = static_cast<UINT>(bytes / 4 < UINT_MAX ? bytes / 4 : UINT_MAX);
    d.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    device_of(the_device)->CreateShaderResourceView(buffer, &d, dest);
}

// gfx10 SQ_BUF_RSRC_WORD3 for a buffer view the title created (udsrc bufword3 / uavword3 of a view token): the
// word gfx10_view_words.h synthesizes, the same value compute hands to compute programs (d3d12_compute.inl): typed
// views only (FORMAT, DST_SEL, OOB_SELECT); raw, structured, constant-buffer and unmapped views: 0.
bool typed_views_off();
std::uint32_t view_buffer_word3(const ViewRecord& record)
{
    if ((record.slot != 18 && record.slot != 19) || typed_views_off()) return 0;
    x2w_startup::BufferViewLayout l;
    if (!x2w_startup::buffer_view_layout_from_desc(record.slot == 19, record.desc, record.desc_bytes, l) ||
        l.kind != x2w_startup::BufferViewKind::typed)
        return 0;
    std::uint32_t v[8];
    x2w_startup::gfx10_buffer_words(l, v);
    return v[3];
}

// Title ranges the GPU writes: translated-draw UAVs and GPU-ordered CopyBufferRegion destinations (slot 15,
// X2W_D3D12_COPY_BUFFER=gpu). Raw srvs overlapping one are bound in place instead of copied at record time, so a
// later draw reads what the GPU stored. Disjoint merged ranges, begin -> end; at most gpu_written_max, the lowest
// other range is dropped beyond that (counted: rt_written_evicted). draw_lock held.
std::map<u64, u64>& gpu_written() { static std::map<u64, u64> m; return m; }
SRWLOCK gpu_written_lock = SRWLOCK_INIT;
constexpr std::size_t gpu_written_max = 16384;
void note_gpu_written_locked(u64 address, u64 bytes);
void note_gpu_written(u64 address, u64 bytes)
{
    AcquireSRWLockExclusive(&gpu_written_lock);
    note_gpu_written_locked(address, bytes);
    ReleaseSRWLockExclusive(&gpu_written_lock);
}
void note_gpu_written_locked(u64 address, u64 bytes)
{
    auto& m = gpu_written();
    u64 begin = address, end = address + bytes;
    auto it = m.upper_bound(begin);
    if (it != m.begin()) {
        const auto before = std::prev(it);
        if (before->second >= begin) {
            begin = before->first;
            if (before->second > end) end = before->second;
            it = m.erase(before);
        }
    }
    while (it != m.end() && it->first <= end) {
        if (it->second > end) end = it->second;
        it = m.erase(it);
    }
    if (m.size() >= gpu_written_max) {
        m.erase(m.begin());
        InterlockedIncrement(&rt_written_evicted);
    }
    m.emplace(begin, end);
}
bool gpu_written_overlaps(u64 address, u64 bytes)
{
    AcquireSRWLockShared(&gpu_written_lock);
    const auto& m = gpu_written();
    auto it = m.lower_bound(address + bytes);
    const bool overlaps = it != m.begin() && (--it)->second > address;
    ReleaseSRWLockShared(&gpu_written_lock);
    return overlaps;
}
std::size_t gpu_written_count()
{
    AcquireSRWLockShared(&gpu_written_lock);
    const std::size_t n = gpu_written().size();
    ReleaseSRWLockShared(&gpu_written_lock);
    return n;
}
// An in-place srv on a window: shader-resource state in this list's tracked states, except when this list already
// holds the window as UAV (the same 256 MiB window often carries the draw's own uav: one state cannot be both;
// it stays UAV, as before this tracking existed).
void window_read_state(Wrapper* self, ID3D12Resource* window)
{
    for (const auto& known : pass_of(self).states)
        if (known.first == window && known.second == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) return;
    require_state(self, window, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
}

void null_uav(D3D12_CPU_DESCRIPTOR_HANDLE dest, int kind)
{
    D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
    if (kind == 1) {
        d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    } else {
        d.Format = DXGI_FORMAT_R32_TYPELESS;
        d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        d.Buffer.NumElements = 1;
        d.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    }
    device_of(the_device)->CreateUnorderedAccessView(nullptr, nullptr, &d, dest);
}

// The 32-byte title descriptor table entry uav j was read from (0: none). words gets its contents.
u64 uav_entry(const Pass& pass, const Translation::Uav& uav, std::uint32_t (&words)[8])
{
    const u64 entry = uav.source == 0 && uav.a >= 0 && uav.a < 16 && uav.b >= 0 ? pass.view_tables[uav.a].record(static_cast<u64>(uav.b), 32) : 0;
    if (!entry || !x2w_local::readable(reinterpret_cast<const void*>(entry), 32)) return 0;
    std::memcpy(words, reinterpret_cast<const void*>(entry), 32);
    return entry;
}

void bind_uav_buffer(const Pass& pass, const Translation& tr, UINT j, D3D12_CPU_DESCRIPTOR_HANDLE dest, std::uint32_t* word3, std::uint32_t* base,
                     ID3D12Resource** bound_window, std::uint32_t* record_stride)
{
    const Translation::Uav& uav = tr.uavs[j];
    std::uint32_t words[8] = {};
    const u64 entry = uav_entry(pass, uav, words);
    u64 address = 0;
    std::size_t bytes = 0;
    const char* from = "none";
    if (uav.source == 2 && uav.a >= 0 && uav.a < 16) {
        address = pass.root_address[uav.a];
        bytes = address ? readable_window(address, 1u << 20) : 0;
        from = "root";
    } else if (entry && words[0] == view_token_magic) {
        // A view the title created: a CBV names its memory, a buffer SRV / UAV a range of its resource's title memory.
        ViewRecord record{};
        AcquireSRWLockShared(&note_lock);
        const bool known = words[1] < view_records().size();
        if (known) record = view_records()[words[1]];
        ReleaseSRWLockShared(&note_lock);
        from = "view";
        if (known) *word3 = view_buffer_word3(record);
        if (known && record.slot == 17 && record.desc_bytes >= 12) {
            std::uint32_t size = 0;
            std::memcpy(&address, record.desc, 8);
            std::memcpy(&size, record.desc + 8, 4);
            bytes = size;
        } else if (known && (record.slot == 18 || record.slot == 19) && record.desc_bytes >= 36) {
            DXGI_FORMAT format{};
            std::uint32_t dimension = 0, count = 0, stride = 0, flags = 0;
            u64 first_element = 0;
            const std::size_t at = record.slot == 18 ? 16 : 8;  // Buffer member of the SRV / UAV description
            std::memcpy(&format, record.desc, 4);
            std::memcpy(&dimension, record.desc + 4, 4);
            std::memcpy(&first_element, record.desc + at, 8);
            std::memcpy(&count, record.desc + at + 8, 4);
            std::memcpy(&stride, record.desc + at + 12, 4);
            std::memcpy(&flags, record.desc + 32, 4);  // Flags: +32 in both descriptions
            Wrapper* resource = as_wrapper(reinterpret_cast<const void*>(record.resource));
            const bool raw = (flags & 1) || format == DXGI_FORMAT_R32_TYPELESS;
            const UINT element = stride ? stride : raw || format == DXGI_FORMAT_UNKNOWN ? 4 : x2w_startup::typed_buffer_format_bytes(format);
            // A structured view's stride, or a typed view's element size; a raw view has no record stride.
            *record_stride = stride ? stride : raw || format == DXGI_FORMAT_UNKNOWN ? 0 : element;
            if (dimension == 1 && element && resource && resource->own().title_memory) {
                address = reinterpret_cast<u64>(resource->own().title_memory) + first_element * element;
                bytes = std::size_t{count} * element;
            }
        }
    } else if (entry) {
        // A hardware buffer descriptor the title wrote itself: base address, stride and record count.
        address = words[0] | (static_cast<u64>(words[1] & 0xffff) << 32);
        const std::uint32_t stride = (words[1] >> 16) & 0x3fff;
        bytes = static_cast<std::size_t>(stride ? stride : 1) * words[2];
        *word3 = words[3];
        *record_stride = stride;
        from = "descriptor";
    }
    if (bytes > (256u << 20)) bytes = 256u << 20;
    ID3D12Resource* window = nullptr;
    u64 offset = 0;
    bool found = address && bytes && desktop_range(address, bytes, &window, &offset);
    // A root UAV has no size: shrink the readable guess until it sits in one window.
    for (std::size_t smaller = bytes; !found && uav.source == 2 && address && smaller > 4096;) {
        smaller = smaller > (64u << 10) ? (64u << 10) : 4096;
        if ((found = desktop_range(address, smaller, &window, &offset))) bytes = smaller;
    }
    const u64 aligned = offset & ~u64{15};
    const char* result = "bound";
    if (!found) {
        result = "NULL (no desktop window)";
        InterlockedIncrement(&rt_uav_buf_unresolved);
        null_uav(dest, 0);
    } else if (aligned != offset && !(tr.uav_base_wanted & (1u << j))) {
        result = "NULL (not 16-byte aligned, no uavbase)";
        InterlockedIncrement(&rt_uav_buf_misaligned);
        null_uav(dest, 0);
    } else {
        *base = static_cast<std::uint32_t>(offset - aligned);
        D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
        d.Format = DXGI_FORMAT_R32_TYPELESS;
        d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        d.Buffer.FirstElement = aligned / 4;
        d.Buffer.NumElements = static_cast<UINT>((bytes + *base + 3) / 4);
        d.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device_of(the_device)->CreateUnorderedAccessView(window, nullptr, &d, dest);
        *bound_window = window;
        note_gpu_written(address, bytes);
        InterlockedIncrement(&rt_uav_buf_bound);
    }
    static LONG shown;
    if (first_calls(shown, 8))
        log("x2w-draw-rt   %016llx uav %u buffer source %d(%d,%d) from %s: entry 0x%llx words %08x %08x %08x %08x -> address 0x%llx, %zu bytes, "
            "window %p +0x%llx: %s", tr.key, j, uav.source, uav.a, uav.b, from, entry, words[0], words[1], words[2], words[3], address, bytes,
            static_cast<void*>(window), offset, result);
}

// Texture uav j (`uav <i> tex2d`): a typed Texture2D UAV in the format and at the mip of the view the title
// created. The draw is skipped, not bound to a mismatched view, when any check fails. draw_lock held.
DXGI_FORMAT typeless_to_float(DXGI_FORMAT f);
bool float_format(DXGI_FORMAT f);
enum UavTextureSkip { tex_ok, tex_unresolved, tex_no_uav_flag, tex_shape, tex_format, tex_typed_store, tex_skip_count };
const char* const uav_texture_skip_names[tex_skip_count] = {"bound", "unresolved view", "texture without ALLOW_UNORDERED_ACCESS",
    "not a single-slice 2D texture at a valid mip", "format not a float/unorm format of the texture", "no typed UAV store for the format"};
static_assert(tex_skip_count == std::size(rt_uav_tex), "one counter per texture uav outcome");

bool typed_store_supported(DXGI_FORMAT format)
{
    static signed char known[256];  // 0 not asked, 1 supported, -1 not
    const auto i = static_cast<unsigned>(format);
    if (i >= 256) return false;
    if (!known[i]) {
        D3D12_FEATURE_DATA_FORMAT_SUPPORT query{format};
        const bool yes = SUCCEEDED(device_of(the_device)->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &query, sizeof(query))) &&
                         (query.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) && (query.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE);
        known[i] = yes ? 1 : -1;
        log("x2w-draw-rt format %d: typed UAV store %s", static_cast<int>(format), yes ? "supported" : "NOT supported");
    }
    return known[i] > 0;
}

UavTextureSkip bind_uav_texture(const Pass& pass, const Translation& tr, UINT j, D3D12_CPU_DESCRIPTOR_HANDLE dest, ID3D12Resource** out)
{
    std::uint32_t words[8] = {};
    const u64 entry = uav_entry(pass, tr.uavs[j], words);
    ViewRecord record{};
    bool known = false;
    if (entry && words[0] == view_token_magic) {
        AcquireSRWLockShared(&note_lock);
        known = words[1] < view_records().size();
        if (known) record = view_records()[words[1]];
        ReleaseSRWLockShared(&note_lock);
    }
    Wrapper* resource = known ? as_wrapper(reinterpret_cast<const void*>(record.resource)) : nullptr;
    ID3D12Resource* texture = resource && resource->own().kind == resource_kind ? static_cast<ID3D12Resource*>(resource->own().real) : nullptr;
    if (!texture || (record.slot != 18 && record.slot != 19) || record.desc_bytes < 24) return tex_unresolved;
    const D3D12_RESOURCE_DESC rd = texture->GetDesc();
    if (!(rd.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) return tex_no_uav_flag;
    DXGI_FORMAT format{};
    std::uint32_t dimension = 0, mip = 0;
    std::memcpy(&format, record.desc, 4);
    std::memcpy(&dimension, record.desc + 4, 4);
    std::memcpy(&mip, record.desc + (record.slot == 19 ? 8 : 16), 4);  // UAV MipSlice / SRV MostDetailedMip
    // TEXTURE2D is 4 in both the SRV and the UAV dimension enumerations.
    if (rd.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || rd.DepthOrArraySize != 1 || rd.SampleDesc.Count != 1 || dimension != 4 ||
        mip >= rd.MipLevels) return tex_shape;
    format = typeless_to_float(desktop_format(format ? format : rd.Format));
    if (!float_format(format) || (format != rd.Format && format != typeless_to_float(rd.Format))) return tex_format;
    if (!typed_store_supported(format)) return tex_typed_store;
    D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
    d.Format = format;
    d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    d.Texture2D.MipSlice = mip;
    device_of(the_device)->CreateUnorderedAccessView(texture, nullptr, &d, dest);
    *out = texture;
    static LONG shown;
    if (first_calls(shown, 4))
        log("x2w-draw-rt   %016llx uav %u texture: entry 0x%llx view %u -> %p %llux%u format %d mip %u: bound", tr.key, j, entry, words[1],
            static_cast<void*>(texture), rd.Width, rd.Height, static_cast<int>(format), mip);
    return tex_ok;
}

struct MeshDumpKeys {
    u64 keys[16] = {};
    unsigned count = 0;
};
const MeshDumpKeys& mesh_dump_keys()
{
    static const MeshDumpKeys list = [] {
        MeshDumpKeys out;
        char value[16 * 20 + 16]{};
        if (!GetEnvironmentVariableA("X2W_MESH_DUMP_INPUTS", value, sizeof(value))) return out;
        for (char* p = value; *p && out.count < 16;) {
            while (*p == ',' || *p == ' ' || *p == ';') ++p;
            char* end = nullptr;
            const u64 key = _strtoui64(p, &end, 16);
            if (end == p) break;
            if (key) out.keys[out.count++] = key;
            p = end;
        }
        log("x2w-mesh X2W_MESH_DUMP_INPUTS: %u key(s)", out.count);
        return out;
    }();
    return list;
}
u64 mesh_dump_key() { return mesh_dump_keys().count ? mesh_dump_keys().keys[0] : 0; }  // non-zero: any key wanted
LONG mesh_dump_claimed[16];              // one dump per listed key
LONG mesh_dump_done;                     // keys dumped
// Any listed key not dumped yet (then the indirect stage reads each mesh call's record, in any frame).
bool mesh_dump_pending() { return mesh_dump_keys().count && mesh_dump_done < static_cast<LONG>(mesh_dump_keys().count); }
// X2W_MESH_DUMP_FROM_FRAME=<n> (default 0): no dump before present n (to skip the start-up pages if needed).
LONG mesh_dump_from_frame()
{
    static const LONG n = [] {
        char v[16]{};
        return GetEnvironmentVariableA("X2W_MESH_DUMP_FROM_FRAME", v, sizeof(v)) ? std::atol(v) : 0L;
    }();
    return n;
}
thread_local std::string mesh_dump_dir;  // non-empty: this call is being dumped (folder with trailing backslash)
bool mesh_dump_record_groups(std::uint32_t (&xyz)[3]);
bool mesh_dump_begin(u64 key)
{
    mesh_dump_dir.clear();
    const MeshDumpKeys& list = mesh_dump_keys();
    unsigned at = 0;
    while (at < list.count && list.keys[at] != key) ++at;
    if (!key || at == list.count || mesh_dump_claimed[at] || frame_number < mesh_dump_from_frame()) return false;
    std::uint32_t xyz[3] = {};
    if (!mesh_dump_record_groups(xyz) || !xyz[0] || !xyz[1] || !xyz[2]) return false;
    if (InterlockedCompareExchange(&mesh_dump_claimed[at], 1, 0) != 0) return false;
    InterlockedIncrement(&mesh_dump_done);
    char log_path[MAX_PATH]{};
    std::string dir;
    if (GetEnvironmentVariableA("X2W_TRAP_LOG", log_path, sizeof(log_path))) {
        dir = log_path;
        const std::size_t slash = dir.find_last_of("\\/");
        dir = slash == std::string::npos ? std::string() : dir.substr(0, slash + 1);
    }
    char name[48];
    std::snprintf(name, sizeof(name), "mesh_dump_%016llx\\", static_cast<unsigned long long>(key));
    dir += name;
    CreateDirectoryA(dir.c_str(), nullptr);
    mesh_dump_dir = dir;
    log("x2w-mesh dump %016llx: inputs of the first call with groups (record %u %u %u, frame %ld%s) to %s", static_cast<unsigned long long>(key),
        xyz[0], xyz[1], xyz[2], frame_number, trace_frame_active ? ", traced" : "", dir.c_str());
    return true;
}
void mesh_dump_file(const char* file, const void* data, std::size_t bytes, const char* note)
{
    if (mesh_dump_dir.empty()) return;
    const std::string path = mesh_dump_dir + file;
    bool written = false;
    if (FILE* f = std::fopen(path.c_str(), "wb")) {
        written = (!bytes || !data || std::fwrite(data, 1, bytes, f) == bytes);
        std::fclose(f);
    }
    if (FILE* index = std::fopen((mesh_dump_dir + "index.txt").c_str(), "ab")) {
        std::fprintf(index, "%s %zu bytes%s: %s\n", file, bytes, written ? "" : " (WRITE FAILED)", note);
        std::fclose(index);
    }
}

thread_local u64 last_draw_key;
thread_local const char* last_draw_why;
bool d16_srvs_ready(const Translation& tr, const Pass& pass, std::uint32_t (&d16_unorm16)[32])
{
    if (!tr.d16_srvs.any()) return true;
    const auto refuse = [&](UINT slot, const char* reason, int detail) {
        static LONG refused;
        if (first_calls(refused, 6))
            log("x2w-draw %016llx srv %u: d16 exact refused (%s %d)", tr.key, slot, reason, detail);
        return false;
    };
    for (UINT i = 0; i < tr.srvs.size(); ++i) {
        if (!tr.d16_srvs.requires_exact(i)) continue;
        const Translation::Srv& srv = tr.srvs[i];
        if (srv.kind < 1 || srv.kind > 4 || srv.source != 0 || srv.a < 0 || srv.a >= 16 || srv.b < 0)
            return refuse(i, "non-table texture binding", srv.kind);
        const auto dimension = srv.kind == 1 ? D3D12_SRV_DIMENSION_TEXTURE2D :
                               srv.kind == 2 ? D3D12_SRV_DIMENSION_TEXTURE2DARRAY :
                               srv.kind == 4 ? D3D12_SRV_DIMENSION_TEXTURE1D : D3D12_SRV_DIMENSION_TEXTURE3D;
        const u64 entry = pass.view_tables[srv.a].record(static_cast<u64>(srv.b), 32);
        std::uint32_t words[8] = {};
        if (!entry || !x2w_local::readable(reinterpret_cast<const void*>(entry), 32))
            return refuse(i, "unreadable table entry", srv.a);
        std::memcpy(words, reinterpret_cast<const void*>(entry), 32);
        if (words[0] != view_token_magic) return refuse(i, "missing view token", 0);
        ViewRecord record{};
        bool known = false;
        AcquireSRWLockShared(&note_lock);
        known = words[1] < view_records().size();
        if (known) record = view_records()[words[1]];
        ReleaseSRWLockShared(&note_lock);
        if (!known || record.slot != 18) return refuse(i, "view slot", static_cast<int>(record.slot));
        Wrapper* resource = as_wrapper(reinterpret_cast<const void*>(record.resource));
        ID3D12Resource* texture = sampled_address_texture(resource);
        if (!texture) return refuse(i, "missing texture resource", 0);
        if (x2w_startup::packed_stand_in_of(texture) != DXGI_FORMAT_UNKNOWN)
            return refuse(i, "packed stand-in resource", static_cast<int>(x2w_startup::packed_stand_in_of(texture)));
        const D3D12_RESOURCE_DESC tex_desc = texture->GetDesc();
        if (srv.kind == 4 && (tex_desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D || tex_desc.DepthOrArraySize != 1))
            return refuse(i, "1D resource mismatch", static_cast<int>(tex_desc.Dimension));
        const auto* original = resource_info(resource);
        DXGI_FORMAT effective = original ? original->console.Format : tex_desc.Format;
        if (record.desc_bytes >= 40) {
            D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
            std::memcpy(&desc, record.desc, 40);
            if (desc.ViewDimension != dimension)
                return refuse(i, "view dimension", static_cast<int>(desc.ViewDimension));
            if (desc.Format != DXGI_FORMAT_UNKNOWN) effective = desc.Format;
        } else if (record.desc_bytes == 0) {
            const bool dim_ok =
                (srv.kind == 1 && tex_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && tex_desc.DepthOrArraySize == 1) ||
                (srv.kind == 2 && tex_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D) ||
                (srv.kind == 3 && tex_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D) ||
                (srv.kind == 4 && tex_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D && tex_desc.DepthOrArraySize == 1);
            if (!dim_ok) return refuse(i, "default view dimension", static_cast<int>(tex_desc.Dimension));
        } else {
            return refuse(i, "truncated view desc", static_cast<int>(record.desc_bytes));
        }
        const auto mode = x2w_startup::d16_srv_mode(effective);
        if (mode == x2w_startup::D16SrvMode::refused)
            return refuse(i, "format", static_cast<int>(effective));
        d16_unorm16[i] = mode == x2w_startup::D16SrvMode::unorm16 ? 1u : 0u;
    }
    return true;
}

// FXC keeps unused entry declarations in reflection. Check actual uses so a
// generated shader which reads the draw's start vertex or instance is refused.
bool indirect_shader_entry(ID3DBlob* blob, const Translation& translation, bool indexed = false)
{
    x2w_startup::ComPtr<ID3D12ShaderReflection> shader;
    if (!blob || FAILED(D3DReflect(blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(shader.GetAddressOf())))) return false;
    D3D12_SHADER_DESC desc{};
    if (FAILED(shader->GetDesc(&desc)) || ((desc.Version >> 4) & 0xf) != 5) return false;
    for (UINT binding = 0; binding < desc.BoundResources; ++binding) {
        D3D12_SHADER_INPUT_BIND_DESC input{};
        if (FAILED(shader->GetResourceBindingDesc(binding, &input))) return false;
        if (input.Type != D3D_SIT_CBUFFER) continue;
        if (!input.Name || std::strcmp(input.Name, "entry") || input.Space || input.BindPoint || input.BindCount != 1) return false;
        auto* buffer = shader->GetConstantBufferByName(input.Name);
        D3D12_SHADER_BUFFER_DESC cb{};
        if (!buffer || FAILED(buffer->GetDesc(&cb))) return false;
        for (UINT i = 0; i < cb.Variables; ++i) {
            D3D12_SHADER_VARIABLE_DESC var{};
            if (FAILED(buffer->GetVariableByIndex(i)->GetDesc(&var))) return false;
            if (!(var.uFlags & D3D_SVF_USED)) continue;
            const auto word = var.StartOffset / sizeof(UINT);
            char name[16];
            std::snprintf(name, sizeof(name), "ud%u", static_cast<unsigned>(word));
            if (var.StartOffset % sizeof(UINT) || var.Size != sizeof(UINT) || word >= translation.uds.size() ||
                !var.Name || std::strcmp(var.Name, name)) return false;
            const auto& ud = translation.uds[word];
            if (indexed && ud.kind == Translation::base_vertex) continue;
            if (ud.kind == Translation::stride && ud.a >= 0 && static_cast<std::size_t>(ud.a) < translation.srvs.size() &&
                (translation.srvs[ud.a].source == 0 || (indexed && translation.srvs[ud.a].source == 1)) &&
                translation.srvs[ud.a].kind == 0) continue;
            if (ud.kind != Translation::root_constant && ud.kind != Translation::rect && ud.kind != Translation::d16_unorm16)
                return false;
        }
    }
    return true;
}

namespace indirect_refusal_code {
constexpr const char* pipeline_or_target_unavailable = "pipeline_or_target_unavailable";
constexpr const char* translation_not_ready = "translation_not_ready";
constexpr const char* d16_srv_not_ready = "d16_srv_not_ready";
constexpr const char* formatted_buffer_not_ready = "formatted_buffer_not_ready";
constexpr const char* missing_guest_root = "missing_guest_root";
constexpr const char* live_compute_pipeline = "live_compute_pipeline";
constexpr const char* unsupported_resource_or_entry = "unsupported_resource_or_entry";
constexpr const char* indexed_rect_unavailable = "indexed_rect_unavailable";
constexpr const char* pipeline_unavailable = "pipeline_unavailable";
constexpr const char* sampler_heap_unavailable = "sampler_heap_unavailable";
constexpr const char* sampled_texture_mismatch = "sampled_texture_mismatch";
constexpr const char* sampled_texture_unbindable = "sampled_texture_unbindable";
constexpr const char* incomplete_vertex_stream = "incomplete_vertex_stream";
constexpr const char* incomplete_table_buffer = "incomplete_table_buffer";
constexpr const char* indexed_validation_failed = "indexed_validation_failed";
}  // namespace indirect_refusal_code

struct TranslatedIndirectDraw {
    ID3D12CommandSignature* signature;
    ID3D12Resource* argument;
    u64 offset;
    IndirectIndexView indices;
    u64 shader_key = 0;
    const char* refusal = "translated pipeline or target unavailable";
    const char* refusal_code = indirect_refusal_code::pipeline_or_target_unavailable;

    // Keep prose refusal text and census code updated together.
    void set_refusal(const char* message, const char* code)
    {
        refusal = message;
        refusal_code = code;
    }
};

struct IndexedValidationPipeline {
    x2w_startup::ComPtr<ID3D12Device> device;
    x2w_startup::ComPtr<ID3D12RootSignature> root;
    x2w_startup::ComPtr<ID3D12PipelineState> pipeline;

    // Lifetime boundary: this record owns the device it was built for, its root signature and its
    // pipeline, from the first validation that asked for them until the module that holds it is
    // unloaded. It is a function-local static in indexed_validation_pipeline() below, so it is destroyed
    // during that module's CRT teardown (startup_stubs/d3d12_teardown.h). On process exit the display
    // driver may already be stopped and releasing a pipeline state then fails inside it (amdxc64, #2572),
    // so the three references are detached and nothing reaches the driver. FreeLibrary keeps the process
    // and the driver alive: no flag, and the members above release their objects normally.
    ~IndexedValidationPipeline()
    {
        if (!x2w_startup::d3d12_process_terminating) return;
        device.Detach();
        root.Detach();
        pipeline.Detach();
    }
};

IndexedValidationPipeline& indexed_validation_pipeline(ID3D12Device* device)
{
    // Hold the compute pipeline for the life of the shim and keep incompatible devices out of the route.
    static IndexedValidationPipeline validation = [device] {
        IndexedValidationPipeline result;
        result.device = device;
        D3D12_ROOT_PARAMETER params[4]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants.Num32BitValues = 8;
        params[1].ParameterType = params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[2].Descriptor.ShaderRegister = 1;
        params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        const D3D12_ROOT_SIGNATURE_DESC desc{4, params};
        x2w_startup::ComPtr<ID3DBlob> signature, code;
        HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, nullptr);
        if (SUCCEEDED(hr)) hr = device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                                            IID_PPV_ARGS(&result.root));
        if (SUCCEEDED(hr)) hr = D3DCompile(x2w_startup::indexed_indirect_guard, sizeof(x2w_startup::indexed_indirect_guard) - 1,
                                           "indexed indirect validation", nullptr, nullptr, "guard", "cs_5_0", 0, 0, &code, nullptr);
        if (SUCCEEDED(hr)) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{};
            pipeline.pRootSignature = result.root.Get();
            pipeline.CS = {code->GetBufferPointer(), code->GetBufferSize()};
            hr = device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&result.pipeline));
        }
        if (FAILED(hr)) log("x2w-render translated indexed argument validation setup failed 0x%08lx", static_cast<unsigned long>(hr));
        return result;
    }();
    return validation;
}

ID3D12Resource* validate_indexed_arguments(Wrapper* self, ID3D12GraphicsCommandList* list, Translation& translation,
                                         const TranslatedIndirectDraw& draw, UINT vertices)
{
    auto* device = device_of(self->own().list_device);
    auto& compute = indexed_validation_pipeline(device);
    if (compute.device.Get() != device || !compute.pipeline) return nullptr;
    if (!translation.indexed_signature) {
        D3D12_INDIRECT_ARGUMENT_DESC fields[2]{};
        fields[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
        fields[0].Constant = {0, static_cast<UINT>(translation.indirect_base_word), 1};
        fields[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
        const D3D12_COMMAND_SIGNATURE_DESC signature{24, 2, fields, 0};
        if (FAILED(device->CreateCommandSignature(&signature, translation.root, IID_PPV_ARGS(&translation.indexed_signature))))
            return nullptr;
    }
    const auto argument_desc = draw.argument->GetDesc();
    const auto index_desc = draw.indices.resource->GetDesc();
    if (argument_desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || index_desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
        (index_desc.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) || (argument_desc.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) ||
        argument_desc.Width > UINT_MAX || index_desc.Width > UINT_MAX || draw.offset > UINT_MAX || draw.indices.offset > UINT_MAX)
        return nullptr;

    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = 28;
    buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    buffer.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    x2w_startup::ComPtr<ID3D12Resource> scratch;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&scratch)))) return nullptr;
    const auto keep = [&] {
        AcquireSRWLockExclusive(&allocator_lock);
        const auto found = list_recordings.find(self);
        bool retained = false;
        if (found != list_recordings.end() && found->second) {
            try {
                found->second->indexed_validation.push_back(scratch);
                retained = true;
            } catch (const std::bad_alloc&) {
            }
        }
        ReleaseSRWLockExclusive(&allocator_lock);
        return retained;
    };
    if (!keep()) return nullptr;

    auto* indices = draw.indices.resource;
    const auto before = [&](ID3D12Resource* resource) {
        for (const auto& state : pass_of(self).states)
            if (state.first == resource) return state.second;
        return D3D12_RESOURCE_STATE_COMMON;
    };
    const auto argument_previous = before(draw.argument);
    const auto index_previous = before(indices);
    require_state(self, draw.argument, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (indices != draw.argument && !draw.indices.upload)
        require_state(self, indices, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    require_state(self, scratch.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const UINT bounds[]{static_cast<UINT>(draw.offset), static_cast<UINT>(argument_desc.Width), static_cast<UINT>(draw.indices.offset),
        draw.indices.view.SizeInBytes, static_cast<UINT>(index_desc.Width), draw.indices.view.Format == DXGI_FORMAT_R16_UINT ? 2u : 4u,
        vertices, 0};
    list->SetPipelineState(compute.pipeline.Get());
    list->SetComputeRootSignature(compute.root.Get());
    list->SetComputeRoot32BitConstants(0, 8, bounds, 0);
    list->SetComputeRootShaderResourceView(1, draw.argument->GetGPUVirtualAddress());
    list->SetComputeRootShaderResourceView(2, indices->GetGPUVirtualAddress());
    list->SetComputeRootUnorderedAccessView(3, scratch->GetGPUVirtualAddress());
    list->Dispatch(1, 1, 1);
    require_state(self, scratch.Get(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    if (draw.argument != indices && argument_previous != D3D12_RESOURCE_STATE_COMMON)
        require_state(self, draw.argument, argument_previous);
    constexpr auto shader_reads = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    if (!draw.indices.upload)
        require_state(self, indices, (index_previous & shader_reads) | D3D12_RESOURCE_STATE_INDEX_BUFFER);

    // The clamp changes the native compute root and invalidates its arguments.
    // In render mode slot 43 is the forwarded compute parameter setter.
    const auto& pass = pass_of(self);
    if (pass.desktop_compute_root) {
        list->SetComputeRootSignature(pass.desktop_compute_root.Get());
        for (UINT parameter = 0; parameter < pass.desktop_compute_upload_srvs.size(); ++parameter) {
            const auto address = pass.desktop_compute_upload_srvs[parameter];
            if (address) list->SetComputeRootShaderResourceView(parameter, address);
        }
    }
    return scratch.Get();
}

struct IndirectBufferView {
    ID3D12Resource* resource = nullptr;
    u64 offset = 0, bytes = 0;
    UINT stride = 0;
    bool upload = false;
};

// For indirect draws, bind the complete descriptor range on a known GPU backing. There is no
// fallback to copying an unrelated or stale title allocation into the draw upload ring.
IndirectBufferView indirect_table_buffer(Wrapper* self, const std::uint32_t* words, bool allow_address_alias = false)
{
    constexpr u64 limit = 16u << 20;
    u64 address = 0, offset = 0, bytes = 0;
    UINT stride = 0;
    const Wrapper* owner = nullptr;
    if (words[0] == view_token_magic) {
        ViewRecord record{};
        AcquireSRWLockShared(&note_lock);
        if (words[1] < view_records().size()) record = view_records()[words[1]];
        ReleaseSRWLockShared(&note_lock);
        if (record.slot == 17 && record.desc_bytes >= 12) {
            UINT size = 0;
            std::memcpy(&address, record.desc, 8);
            std::memcpy(&size, record.desc + 8, 4);
            bytes = size;
        } else if (record.slot == 18 && record.desc_bytes >= 40) {
            D3D12_SHADER_RESOURCE_VIEW_DESC view{};
            std::memcpy(&view, record.desc, 40);
            owner = as_wrapper(reinterpret_cast<const void*>(record.resource));
            if (!owner || owner->own().kind != resource_kind || !owner->own().real ||
                view.ViewDimension != D3D12_SRV_DIMENSION_BUFFER) return {};
            const bool raw = view.Buffer.Flags == D3D12_BUFFER_SRV_FLAG_RAW && !view.Buffer.StructureByteStride &&
                view.Format == DXGI_FORMAT_R32_TYPELESS;
            const bool structured = view.Buffer.Flags == D3D12_BUFFER_SRV_FLAG_NONE && view.Buffer.StructureByteStride &&
                view.Format == DXGI_FORMAT_UNKNOWN;
            const bool typed = view.Buffer.Flags == D3D12_BUFFER_SRV_FLAG_NONE && !view.Buffer.StructureByteStride;
            const UINT element = raw ? 4 : structured ? view.Buffer.StructureByteStride :
                typed ? x2w_startup::typed_buffer_format_bytes(view.Format) : 0;
            if (!element || view.Buffer.FirstElement > UINT64_MAX / element) return {};
            offset = view.Buffer.FirstElement * element;
            bytes = u64{view.Buffer.NumElements} * element;
            stride = raw ? 0 : element;
            if (const auto base = reinterpret_cast<u64>(owner->own().title_memory)) {
                if (offset > UINT64_MAX - base) return {};
                address = base + offset;
            }
        }
    } else {
        address = words[0] | (u64{words[1] & 0xffff} << 32);
        stride = (words[1] >> 16) & 0x3fff;
        bytes = u64{stride ? stride : 1} * words[2];
    }
    if (!bytes || bytes > limit || bytes % 4) return {};
    ID3D12Resource* native = nullptr;
    bool upload = false;
    Wrapper* device = self->own().list_device;
    if (address) {
        if (const auto allocation = list_upload_range(self, address, bytes); allocation.resource) {
            if (!owner) return {allocation.resource, allocation.offset, bytes, stride, true};
            return {};
        }
        const auto placed = placed_buffer_range(device, address, bytes);
        if (placed.reject || (owner && owner != placed.owner)) return {};
        if (!placed.owner) {
            const auto alias = allow_address_alias && !list_upload_range(self, address, 1).resource
                ? indirect_address_alias(device, address, bytes) : IndirectAddressAlias{};
            return {alias.resource, alias.offset, bytes, stride};
        }
        owner = placed.owner;
        native = static_cast<ID3D12Resource*>(owner->own().real);
        offset = address - reinterpret_cast<u64>(owner->own().title_memory);
        if (!placed.alias && (!native || !list_copied_gpu_range(self, native, offset, bytes))) return {};
    } else if (owner && !owner->own().title_memory) {
        native = static_cast<ID3D12Resource*>(owner->own().real);
        D3D12_HEAP_PROPERTIES props{};
        D3D12_HEAP_FLAGS flags{};
        if (FAILED(native->GetHeapProperties(&props, &flags)) ||
            (props.Type != D3D12_HEAP_TYPE_DEFAULT && props.Type != D3D12_HEAP_TYPE_UPLOAD)) return {};
        upload = props.Type == D3D12_HEAP_TYPE_UPLOAD;
    }
    if (!native || offset % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT ||
        !InterlockedCompareExchange(&owner->own().refs, 0, 0)) return {};
    const auto* info = resource_info(owner);
    if (!info || info->initial_state != D3D12_RESOURCE_STATE_COMMON) return {};
    x2w_startup::ComPtr<ID3D12Device> native_device;
    if (FAILED(native->GetDevice(IID_PPV_ARGS(native_device.GetAddressOf()))) || native_device.Get() != device_of(device)) return {};
    const auto desc = native->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || (desc.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) ||
        offset > desc.Width || bytes > desc.Width - offset) return {};
    return {native, offset, bytes, stride, upload};
}

struct FormattedBufferDescriptor {
    u64 entry = 0;
    std::uint32_t words[8] = {};
};

bool formatted_buffers_ready(const Translation& tr, const Pass& pass, FormattedBufferDescriptor (&descriptors)[draw_srv_limit])
{
    const auto refuse = [&](int slot, const char* reason, std::uint32_t detail = 0) {
        InterlockedIncrement(&formatted_draws_refused);
        static LONG refused;
        if (first_calls(refused, 6))
            log("x2w-draw %016llx srv %d: formatted buffer refused (%s %u)", tr.key, slot, reason, detail);
        return false;
    };
    for (const auto& ud : tr.uds) {
        if (ud.kind != Translation::desc_word3) continue;
        if (ud.a < 0 || ud.a >= draw_srv_limit || static_cast<std::size_t>(ud.a) >= tr.srvs.size())
            return refuse(ud.a, "invalid SRV index");
        auto& descriptor = descriptors[ud.a];
        if (descriptor.entry) continue;
        const auto& srv = tr.srvs[ud.a];
        if (srv.kind != 0 || srv.source != 0 || srv.a < 0 || srv.a >= 16 || srv.b < 0)
            return refuse(ud.a, "missing hardware descriptor");
        const u64 entry = pass.view_tables[srv.a].record(static_cast<u64>(srv.b), sizeof(descriptor.words));
        if (!entry || !x2w_local::readable(reinterpret_cast<const void*>(entry), sizeof(descriptor.words)))
            return refuse(ud.a, "missing hardware descriptor");
        std::memcpy(descriptor.words, reinterpret_cast<const void*>(entry), sizeof(descriptor.words));
        if (descriptor.words[0] == view_token_magic) {
            // A typed buffer view the title created: the driver builds word 3 from its format and component mapping.
            ViewRecord record{};
            AcquireSRWLockShared(&note_lock);
            const bool known = descriptor.words[1] < view_records().size();
            if (known) record = view_records()[descriptor.words[1]];
            ReleaseSRWLockShared(&note_lock);
            if (!known || record.slot != 18 || record.desc_bytes < 40)
                return refuse(ud.a, "missing hardware descriptor");
            D3D12_SHADER_RESOURCE_VIEW_DESC view;
            std::memcpy(&view, record.desc, 40);
            if (const char* reason = x2w_startup::view_word3(view, descriptor.words[3])) return refuse(ud.a, reason);
            descriptor.entry = entry;
            continue;
        }
        const auto format = x2w_startup::formatted_buffer_code(descriptor.words[3]);
        if (!x2w_startup::formatted_buffer_supported(format))
            return refuse(ud.a, "unsupported FORMAT", format);
        if (const char* reason = x2w_startup::descriptor_word3_refusal(descriptor.words[3])) return refuse(ud.a, reason);
        descriptor.entry = entry;
    }
    return true;
}

bool view_record_words(const ViewRecord& record, std::uint32_t out[8], bool* image, bool* unmapped);  // d3d12_compute.inl

LONG rt_descword_title, rt_descword_view, rt_descword_null;
std::uint32_t draw_descword_value(const Pass& pass, int param, int offset, u64 key)
{
    const u64 entry = param >= 0 && param < 16 && offset >= 0 ? pass.view_tables[param].record(static_cast<u64>(offset) & ~31ull, 32) : 0;
    const int word = (offset & 31) / 4;
    std::uint32_t words[8] = {}, value = 0;
    const char* what = "table not bound or entry out of range";
    if (entry && x2w_local::readable(reinterpret_cast<const void*>(entry), 32)) {
        std::memcpy(words, reinterpret_cast<const void*>(entry), 32);
        if (words[0] == refused_view_magic) {
            what = "refused-view marker: 0";  // the stand-in's marker words are not the console's descriptor
        } else if (words[0] != view_token_magic) {
            value = words[word];
            what = "title word";
        } else {
            ViewRecord record{};
            AcquireSRWLockShared(&note_lock);
            const bool known = words[1] < view_records().size();
            if (known) record = view_records()[words[1]];
            ReleaseSRWLockShared(&note_lock);
            std::uint32_t out[8] = {};
            bool image = false, unmapped = false;
            if (known && view_record_words(record, out, &image, &unmapped)) {
                value = out[word];
                what = image ? (unmapped ? "T# word (no gfx10 format: FORMAT 0)" : "T# word") : "V# word";
            } else {
                what = "unknown view: 0";
            }
        }
    } else if (entry) {
        what = "entry unreadable: 0";
    }
    InterlockedIncrement(what[0] == 't' && what[1] == 'i' ? &rt_descword_title : what[1] == '#' ? &rt_descword_view : &rt_descword_null);
    static LONG shown;
    if (first_calls(shown, 4))
        log("x2w-draw-rt %016llx descword %d %d: entry 0x%llx %s = 0x%08x", key, param, offset, entry, what, value);
    return value;
}

bool draw_translated(Wrapper* self, bool indexed, UINT count, UINT instances, UINT start, INT base_vertex, UINT start_instance,
                     TranslatedIndirectDraw* indirect)
{
    auto* list = recording(self);
    Pass& pass = pass_of(self);
    const u64 ring_refusals_at_start = ring_cap_refusals;
    const bool mesh = mesh_issue_active(self);
    const bool rect = !mesh && pass.topology == 0x11;
    // Upstream #2259: an indexed rectangle list is drawn only through an indirect draw (geometry-stage expansion).
    const bool indexed_rect = rect && indexed && indirect;
    last_draw_key = 0;
    last_draw_why = nullptr;
    if (list && pass.pipeline && !pass.targets_bound) InterlockedIncrement(&draws_without_targets);
    if (!list || !pass.pipeline || !pass.targets_bound || !(mesh || rect || pass.topology == 4 || pass.topology == 5) || (rect && indexed && !indexed_rect)) {
        last_draw_why = !list ? "not recording" : !pass.pipeline ? "no pipeline" : !pass.targets_bound ? "no targets"
                      : (rect && indexed && !indexed_rect) ? "indexed rectangle list" : "topology";
        return false;
    }
    lock_draw(100000 + __LINE__);
    report_draw_rt();
    static const bool pso_locked = x2w_local::env_on("X2W_DRAW_PSO_LOCKED");
    BoundPipeline* bound = nullptr;
    Translation* tr = nullptr;
    ID3D12PipelineState* pipeline = nullptr;
    for (int attempt = 0;; ++attempt) {
        // A second attempt finds the pipeline build_pipeline made (pipeline_for installs it).
        bound = translation_for(pass.pipeline);
        tr = bound ? translation_for_targets(bound->translation, pass) : nullptr;
        PsoBuild build;
        pipeline = tr && tr->state == 1 && ensure_draw_heap(self->own().list_device)
                       ? pipeline_for(*bound, *tr, pass, pso_locked || attempt ? nullptr : &build, indexed_rect) : nullptr;
        if (pipeline || !build.bound) break;
        ReleaseSRWLockExclusive(&draw_lock);
        if (draw_async()) {
            queue_pso_build(build);
            last_draw_key = tr->key;
            last_draw_why = "desktop pipeline compiling (X2W_DRAW_ASYNC)";
            return false;
        }
        build_pipeline(build);
        lock_draw(100000 + __LINE__);
    }
    if (bound && bound->translation) last_draw_key = bound->translation->key;
    // Upstream indirect draws (#2232 .. #2262): reached only through draw_translated(..., indirect), the
    // X2W_TRANSLATED_INDIRECT path in d3d12_xs_device.cpp. A plain draw passes nullptr and skips all of it.
    if (indirect && tr) indirect->shader_key = tr->key;
    std::uint32_t d16_unorm16[32] = {};
    if (tr && tr->state == 1 && !d16_srvs_ready(*tr, pass, d16_unorm16)) {
        last_draw_why = "d16 exact srv refused";
        ReleaseSRWLockExclusive(&draw_lock);
        return false;
    }
    FormattedBufferDescriptor formatted_buffers[draw_srv_limit] = {};  // upstream #2138 (udsrc descword3)
    if (tr && tr->state == 1 && !formatted_buffers_ready(*tr, pass, formatted_buffers)) {
        last_draw_why = "formatted buffer refused";
        ReleaseSRWLockExclusive(&draw_lock);
        return false;
    }
    if (indirect && (!bound || !tr || tr->state != 1)) {
        ReleaseSRWLockExclusive(&draw_lock);
        return false;
    }
    if (indirect) {
        if (!bound->guest_root || bound->guest_root != pass.root) {
            indirect->set_refusal("translated draw requires the guest pipeline's root signature",
                                  indirect_refusal_code::missing_guest_root);
            ReleaseSRWLockExclusive(&draw_lock);
            return false;
        }
        if (indexed && pass.compute_pipeline) {
            indirect->set_refusal("indexed validation cannot replace a live guest compute pipeline",
                                  indirect_refusal_code::live_compute_pipeline);
            ReleaseSRWLockExclusive(&draw_lock);
            return false;
        }
        auto& entry = indexed ? tr->indexed_indirect_entry : tr->independent_indirect_entry;
        if (!entry.has_value()) {
            bool found_base = !indexed;
            if (indexed) {
                for (std::size_t i = 0; i < tr->uds.size(); ++i) {
                    if (tr->uds[i].kind != Translation::base_vertex) continue;
                    if (found_base) { found_base = false; break; }
                    found_base = true;
                    tr->indirect_base_word = static_cast<int>(i);
                }
            }
            entry = found_base && indirect_shader_entry(tr->vs, *tr, indexed) && indirect_shader_entry(tr->ps, *tr, indexed) &&
                std::all_of(tr->srvs.begin(), tr->srvs.end(), [indexed](const Translation::Srv& srv) {
                    return srv.source == 0 || (indexed && srv.kind == 0 && srv.source == 1 && srv.a == 0);
                });
        }
        if (!*entry || tr->samplers.heap) {
            indirect->set_refusal("translated draw uses an unsupported resource or draw-dependent entry value",
                                  indirect_refusal_code::unsupported_resource_or_entry);
            ReleaseSRWLockExclusive(&draw_lock);
            return false;
        }
    }
    if (indexed_rect && !pipeline) indirect->refusal = "translated indexed rectangle geometry unavailable";
    if (!pipeline || (tr->samplers.heap && !prepare_draw_sampler_heap(pass, list, tr->samplers.reserved_fallback))) {
        last_draw_why = !bound || !bound->translation ? "no translation record"
                      : !tr ? "no translation for the bound targets (rtv variant)"
                      : tr->state != 1 ? "no translation" : !pipeline ? "no desktop pipeline" : "sampler heap";
        ReleaseSRWLockExclusive(&draw_lock);
        return false;
    }
    TitleHeapView heap_view;
    if (tr->heap) {
        const char* reason = draw_heap_increment != 32 ? "descriptor increment is not 32" : draw_title_heap(pass, heap_view);
        if (!reason) {
            fill_views(self);
            upload_heap_textures(self);
            if (!vinfo_buffer) reason = "no per-view info buffer";
            else if (tr->heapwords && !ensure_vdesc()) reason = "no view descriptor (vdesc) buffer";
        }
        if (reason) {
            InterlockedIncrement(&rt_heap_skipped);
            static LONG shown;
            if (first_calls(shown, 4)) log("x2w-draw-rt %016llx: heap draw skipped: %s", tr->key, reason);
            last_draw_why = reason;
            ReleaseSRWLockExclusive(&draw_lock);
            return false;
        }
    }
    const UINT n = static_cast<UINT>(tr->srvs.size());
    const UINT m = static_cast<UINT>(tr->uavs.size());
    const UINT hv = tr->heap ? (tr->heapwords ? 3u : 2u) : 0u;
    UINT first = static_cast<UINT>(InterlockedExchangeAdd(&draw_heap_cursor, static_cast<LONG>(n + m + hv))) % draw_heap_size;
    if (first + n + m + hv > draw_heap_size) {
        first = 0;
        InterlockedExchange(&draw_heap_cursor, static_cast<LONG>(n + m + hv));
    }
    std::uint32_t uav_word3[8] = {}, uav_bases[8] = {}, uav_strides[8] = {};
    ID3D12Resource* uav_textures[8] = {};
    ID3D12Resource* uav_windows[8] = {};
    UavTextureSkip skip = tex_ok;
    for (UINT j = 0; j < m && skip == tex_ok; ++j) {
        const D3D12_CPU_DESCRIPTOR_HANDLE dest{draw_heap->GetCPUDescriptorHandleForHeapStart().ptr + SIZE_T{first + n + j} * draw_heap_increment};
        if (tr->uavs[j].kind == 0) bind_uav_buffer(pass, *tr, j, dest, &uav_word3[j], &uav_bases[j], &uav_windows[j], &uav_strides[j]);
        else skip = bind_uav_texture(pass, *tr, j, dest, &uav_textures[j]);
        if (tr->uavs[j].kind == 1) InterlockedIncrement(&rt_uav_tex[skip]);
    }
    // Windows bound as UAV are tracked in this list's states (moved back to COMMON at Close), so a GPU copy or an
    // in-place srv after this draw names the right before-state.
    if (skip == tex_ok)
        for (ID3D12Resource* w : uav_windows)
            if (w) require_state(self, w, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (skip != tex_ok) {
        static LONG shown[tex_skip_count];
        if (first_calls(shown[skip], 2)) log("x2w-draw-rt %016llx: draw skipped: texture uav: %s", tr->key, uav_texture_skip_names[skip]);
        last_draw_why = uav_texture_skip_names[skip];
        ReleaseSRWLockExclusive(&draw_lock);
        return false;
    }
    const bool split = draw_lock_split() && !indirect;
    if (split) ReleaseSRWLockExclusive(&draw_lock);
    std::uint32_t strides[draw_srv_limit] = {}, word3[draw_srv_limit] = {};
    // Only the vertices this draw can reference are copied: [vertex_first, vertex_first + vertex_count).
    // The shader adds the "base vertex" entry value to the vertex index, so that value is biased by
    // -vertex_first and each stream's view starts at vertex_first.
    const UINT index_bytes = pass.ib.Format == DXGI_FORMAT_R16_UINT ? 2 : 4;
    const u64 index_address = pass.ib.BufferLocation + u64{start} * index_bytes;
    const std::size_t index_total = std::size_t{count} * index_bytes;
    const bool indices_ok = indexed && !indirect && pass.ib.BufferLocation && readable_window(index_address, index_total) >= index_total;
    INT vertex_first = indexed ? 0 : static_cast<INT>(start);
    UINT vertex_count = indexed ? 0 : count;
    if (indices_ok) {
        std::uint32_t low = ~0u, high = 0;
        for (UINT k = 0; k < count; ++k) {
            const std::uint32_t value = index_bytes == 2 ? reinterpret_cast<const std::uint16_t*>(index_address)[k]
                                                         : reinterpret_cast<const std::uint32_t*>(index_address)[k];
            if (value < low) low = value;
            if (value > high) high = value;
        }
        if (low <= high) {
            vertex_first = base_vertex + static_cast<INT>(low);
            vertex_count = high - low + 1;
        }
    }
    if (vertex_first < 0) vertex_first = 0;
    bool dumping = false;
    if (mesh && mesh_dump_pending()) {  // any frame (the first call with groups of each listed key)
        SplitRelock relock(split, 100000 + __LINE__);
        dumping = mesh_dump_begin(tr->key);
    }
    UINT indexed_vertices = UINT_MAX;
    for (UINT i = 0; i < n; ++i) {
        const Translation::Srv& srv = tr->srvs[i];
        const D3D12_CPU_DESCRIPTOR_HANDLE dest{draw_heap->GetCPUDescriptorHandleForHeapStart().ptr + SIZE_T{first + i} * draw_heap_increment};
        u64 address = 0, offset = 0;
        std::size_t bytes = 0;
        PlacedVertexView placed{};
        if (srv.source == 1 && srv.a >= 0 && srv.a < 8) {
            placed = pass.placed_vb[srv.a];
            address = pass.vb[srv.a].BufferLocation;
            bytes = pass.vb[srv.a].SizeInBytes;
            strides[i] = pass.vb[srv.a].StrideInBytes;
            const u64 skip = u64{static_cast<UINT>(vertex_first)} * strides[i];
            if (strides[i] && vertex_count && skip < bytes) {
                const u64 wanted = u64{vertex_count} * strides[i];
                address += skip;
                bytes = static_cast<std::size_t>(wanted < bytes - skip ? wanted : bytes - skip);
            }
        } else if (srv.source == 2 && srv.a < 16) {
            address = pass.root_address[srv.a];
            bytes = readable_window(address, 0x10000);
        }
        const auto& formatted = formatted_buffers[i];
        const u64 entry = formatted.entry ? formatted.entry :
            (srv.source == 0 && srv.a >= 0 && srv.a < 16 && srv.b >= 0 ?
                pass.view_tables[srv.a].record(static_cast<u64>(srv.b), 32) : 0);
        std::uint32_t words[8] = {};
        const bool entry_ok = entry && x2w_local::readable(reinterpret_cast<const void*>(entry), 32);
        if (entry_ok) std::memcpy(words, reinterpret_cast<const void*>(entry), 32);
        if (entry_ok && words[0] != view_token_magic) word3[i] = words[3];  // run-time buffer format (udsrc bufword3)
        bool buffer_token_known = false;
        if (srv.kind >= 1 && srv.kind <= 4) {
            const auto dimension = srv.kind == 1 ? D3D12_SRV_DIMENSION_TEXTURE2D :
                                   srv.kind == 2 ? D3D12_SRV_DIMENSION_TEXTURE2DARRAY :
                                   srv.kind == 4 ? D3D12_SRV_DIMENSION_TEXTURE1D : D3D12_SRV_DIMENSION_TEXTURE3D;
            if (entry_ok) note_rewritten_marker(tr->key, i, entry, words);
            SplitRelock relock(split, 100000 + __LINE__);  // the rest of the texture bind (memcut revive, uploads, counters)
            if (entry_ok && words[0] != view_token_magic) resolve_heap_index(tr->key, i, entry, words);
            // A texture: the title's descriptor memory carries the id of the view it created there.
            ViewRecord record{};
            bool known = false;
            if (words[0] == view_token_magic) {
                AcquireSRWLockShared(&note_lock);
                known = words[1] < view_records().size();
                if (known) record = view_records()[words[1]];
                ReleaseSRWLockShared(&note_lock);
            }
            Wrapper* resource = known ? as_wrapper(reinterpret_cast<const void*>(record.resource)) : nullptr;
            D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
            desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            desc.ViewDimension = dimension;
            desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            desc.Texture2D.MipLevels = 1;  // shared MostDetailedMip and MipLevels in all supported texture views
            if (srv.kind == 2) desc.Texture2DArray.ArraySize = 1;
            ID3D12Resource* texture = sampled_address_texture(resource);
            if (texture && record.desc_bytes >= 40) {
                std::memcpy(&desc, record.desc, 40);
                desc.Format = desktop_format(desc.Format);
            }
            if (texture && x2w_startup::packed_target_format(desc.Format)) {
                // The view recorded from the title names a packed format: follow a stand-in texture,
                // and bind nothing for a texture the view does not fit (packed_target_format.h).
                const auto actual = texture->GetDesc().Format;
                const auto verdict = x2w_startup::packed_view(desc.Format, actual, x2w_startup::packed_stand_in_of(texture));
                if (verdict == x2w_startup::PackedView::stand_in) desc.Format = actual;
                if (verdict == x2w_startup::PackedView::refused) {
                    static LONG refused;
                    if (first_calls(refused, 6))
                        log("x2w-draw   %016llx srv %u: packed format %d on a texture of format %d that does not stand in for it; binding null",
                            tr->key, i, desc.Format, actual);
                    texture = nullptr;
                    desc = {};
                    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
                    desc.ViewDimension = dimension;
                    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                    desc.Texture2D.MipLevels = 1;
                    if (srv.kind == 2) desc.Texture2DArray.ArraySize = 1;
                }
            }
            if (indirect && (!texture || record.slot != 18 || texture == pass.target0)) {
                indirect->set_refusal("translated draw requires a matching sampled texture distinct from the target",
                                      indirect_refusal_code::sampled_texture_mismatch);
                ReleaseSRWLockExclusive(&draw_lock);
                return false;
            }
            if (texture && !upload_texture(self, resource, texture)) texture = nullptr;
            if (texture) {
                require_state(self, texture, sampled_state(self, texture));
                if (resource && !resource->own().tiled_address_srv) retire_hold(self, resource);
            }
            const bool had_texture = texture != nullptr;
            bool mismatched = false;
            static LONG shown;
            if (first_calls(shown, 6))
                log("x2w-draw   %016llx srv %u texture: table entry 0x%llx words %08x %08x -> view %s, resource %p, desc %u bytes format %d",
                    tr->key, i, entry, words[0], words[1], known ? "known" : "UNKNOWN", static_cast<void*>(texture), record.desc_bytes, desc.Format);
            if (trace_frame_active) {
                const D3D12_RESOURCE_DESC td = texture ? texture->GetDesc() : D3D12_RESOURCE_DESC{};
                log("x2w-frame srv %016llx %u texture %p %llux%u fmt %d (view fmt %d)", tr->key, i, static_cast<void*>(texture),
                    static_cast<unsigned long long>(td.Width), td.Height, td.Format, desc.Format);
            }
            const bool wrong_1d_resource = texture && srv.kind == 4 &&
                (texture->GetDesc().Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D || texture->GetDesc().DepthOrArraySize != 1);
            if (texture && ((record.desc_bytes >= 40 && desc.ViewDimension != dimension) || wrong_1d_resource)) {
                static LONG mismatch;
                if (first_calls(mismatch, 6)) log("x2w-draw   %016llx srv %u: view dimension %d does not match the shader; binding null", tr->key, i, desc.ViewDimension);
                texture = nullptr;
                mismatched = true;
                desc.ViewDimension = dimension;
                desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
                desc.Texture2DArray = {};
                desc.Texture2D.MipLevels = 1;
                if (srv.kind == 2) desc.Texture2DArray.ArraySize = 1;
            }
            if (indirect && !texture) {
                indirect->set_refusal("translated draw could not bind its sampled texture",
                                      indirect_refusal_code::sampled_texture_unbindable);
                ReleaseSRWLockExclusive(&draw_lock);
                return false;
            }
            x2w_local::BindingFacts facts;
            facts.texture = facts.from_table = true;
            facts.entry_readable = entry_ok;
            facts.token_magic = words[0] == view_token_magic;
            facts.token_known = known;
            facts.resource_wrapper = had_texture;
            facts.dimension_matches = !mismatched;
            const auto cause = x2w_local::classify_null_binding(facts);
            count_null_binding(cause);
            if (cause == x2w_local::NullCause::unknown_token && token_null_census.note(tr->key)) {
                char line[256];
                x2w_startup::report_token_entry(line, sizeof(line), tr->key, i, entry, words);
                log("%s", line);
            }
            census_texture_binding(resource, texture, x2w_local::classify_null_binding(facts));
            note_null_texture(tr->key, i, entry, words, x2w_local::classify_null_binding(facts));
            D3D12_SHADER_RESOURCE_VIEW_DESC depth_default{};
            const bool no_desc = texture && record.desc_bytes < 40;
            const bool depth_desc = no_desc && default_depth_srv(texture, depth_default);
            if (texture && !no_desc) fix_srv(desc, texture);
            device_of(the_device)->CreateShaderResourceView(texture, depth_desc ? &depth_default : no_desc ? nullptr : &desc, dest);
            continue;
        }
        if (srv.source == 0 && entry_ok) note_rewritten_marker(tr->key, i, entry, words);
        const bool buffer_link = srv.source == 0 && entry_ok && free_link_words(words);
        if (buffer_link) {
            InterlockedIncrement(words[0] == refused_view_magic ? &buffer_refused_views : &buffer_links);
            if (buffer_heap_index_on()) {
                static LONG link_lines;
                if (trace_frame_active || first_calls(link_lines, 32)) {
                    SplitRelock relock(split, 100000 + __LINE__);  // reads heap shadow arrays outside note_lock
                    log_buffer_link(tr->key, i, entry, words);
                }
                InterlockedIncrement(&buffer_links_nulled);
                std::memset(words, 0, sizeof(words));
                word3[i] = 0;
            }
        }
        if (indirect) {
            IndirectBufferView view{};
            if (srv.source == 1 && indexed) {
                const auto& vb = pass.vb[0];
                if (vb.StrideInBytes && vb.StrideInBytes <= 0x3fff && vb.SizeInBytes % vb.StrideInBytes == 0 &&
                    vb.BufferLocation <= 0xffffffffffffull) {
                    const std::uint32_t stream[]{static_cast<UINT>(vb.BufferLocation),
                        static_cast<UINT>(vb.BufferLocation >> 32) | vb.StrideInBytes << 16, vb.SizeInBytes / vb.StrideInBytes, 0};
                    view = indirect_table_buffer(self, stream, true);
                    indexed_vertices = vb.SizeInBytes / vb.StrideInBytes;
                }
            } else if (entry_ok) {
                view = indirect_table_buffer(self, words);
            }
            if (!view.resource || view.resource == indirect->argument || view.offset % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT) {
                indirect->set_refusal(
                    srv.source == 1 ? "translated indexed draw requires a complete GPU-backed vertex stream"
                                    : "translated draw requires a complete GPU-backed table buffer range",
                    srv.source == 1 ? indirect_refusal_code::incomplete_vertex_stream
                                    : indirect_refusal_code::incomplete_table_buffer);
                ReleaseSRWLockExclusive(&draw_lock);
                return false;
            }
            if (!view.upload)
                require_state(self, view.resource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            strides[i] = view.stride;
            buffer_view(dest, view.resource, view.offset, static_cast<std::size_t>(view.bytes));
            continue;
        }
        if (srv.source == 0 && words[0] == view_token_magic && entry) {
            // A view the title created: a constant buffer view names its memory directly; a buffer
            // shader-resource view names a range of its resource's title memory.
            ViewRecord record{};
            AcquireSRWLockShared(&note_lock);
            buffer_token_known = words[1] < view_records().size();
            if (buffer_token_known) record = view_records()[words[1]];
            ReleaseSRWLockShared(&note_lock);
            if (buffer_token_known) word3[i] = view_buffer_word3(record);
            if (record.slot == 17 && record.desc_bytes >= 12) {
                std::uint32_t size = 0;
                std::memcpy(&address, record.desc, 8);
                std::memcpy(&size, record.desc + 8, 4);
                bytes = size;
            } else if (record.slot == 18 && record.desc_bytes >= 40) {
                D3D12_SHADER_RESOURCE_VIEW_DESC view;
                std::memcpy(&view, record.desc, 40);
                Wrapper* resource = as_wrapper(reinterpret_cast<const void*>(record.resource));
                if (view.ViewDimension == D3D12_SRV_DIMENSION_BUFFER && resource && resource->own().title_memory) {
                    const bool typed_view = !view.Buffer.StructureByteStride && view.Format != DXGI_FORMAT_R32_TYPELESS;
                    const UINT element = view.Buffer.StructureByteStride ? view.Buffer.StructureByteStride
                                         : typed_view ? x2w_startup::typed_buffer_format_bytes(view.Format) : 4;
                    address = reinterpret_cast<u64>(resource->own().title_memory) + view.Buffer.FirstElement * element;
                    bytes = std::size_t{view.Buffer.NumElements} * element;
                    strides[i] = typed_view ? element : view.Buffer.StructureByteStride;
                }
            }
        }
        if (srv.source == 0 && words[0] != view_token_magic && entry) {
            // A hardware buffer descriptor the title wrote itself: base address, stride and record count.
            address = words[0] | (static_cast<u64>(words[1] & 0xffff) << 32);
            strides[i] = (words[1] >> 16) & 0x3fff;
            bytes = static_cast<std::size_t>(strides[i] ? strides[i] : 1) * words[2];
        }
        const std::size_t full_bytes = bytes;
        if (mesh && trace_frame_active) record_trace_mesh_read(tr->key, i, srv.source, srv.a, srv.b, address, full_bytes);
        if (bytes > (16u << 20)) bytes = 16u << 20;
        if (dumping) {
            char file[16], note[240];
            std::snprintf(file, sizeof(file), "srv%u.bin", i);
            // A root srv has no size (the binding takes up to 64 KiB): the dump takes the readable window up to 16 MiB.
            // Any range is cut to its readable part (the note says so) instead of dropped.
            const std::size_t wanted = srv.source == 2 ? (16u << 20) : full_bytes > (256u << 20) ? (256u << 20) : full_bytes;
            const std::size_t dump_bytes = address && wanted ? static_cast<std::size_t>(readable_window(address, wanted)) : 0;
            const bool readable_range = dump_bytes != 0;
            std::snprintf(note, sizeof(note), "srv %u source %d(%d,%d) entry 0x%llx words %08x %08x %08x %08x address 0x%llx stride %u bound %zu%s%s%s", i,
                          srv.source, srv.a, srv.b, static_cast<unsigned long long>(entry), words[0], words[1], words[2], words[3],
                          static_cast<unsigned long long>(address), strides[i], bytes, readable_range ? "" : " (range not readable: no bytes)",
                          placed.reject ? " (placed: rejected, bound null)" : "",
                          srv.source == 2 ? " (root srv: readable window up to 16 MiB)"
                          : dump_bytes && dump_bytes < wanted ? " (cut to the readable part)"
                          : dump_bytes != bytes ? " (binding clamped to 16 MiB; dump has the full range up to 256 MiB)" : "");
            SplitRelock relock(split, 100000 + __LINE__);
            mesh_dump_file(file, readable_range ? reinterpret_cast<const void*>(address) : nullptr, readable_range ? dump_bytes : 0, note);
        }
        if (const int r = range_watch_hit(address, bytes); r >= 0) {
            SplitRelock relock(split, 100000 + __LINE__);
            static LONG shown;
            if (first_calls(shown, 16) && range_watch_line(r))
                log("x2w-watch 0x%llx: draw %016llx srv %u reads 0x%llx (%zu bytes) frame %ld", range_watch_address(r), tr->key, i, address, bytes,
                    frame_number);
            range_watch_sample("draw-bind", tr->key);
        }
        // A buffer binding that ends up without a readable range counts as a null binding with its cause.
        const auto count_buffer = [&](bool bound) {
            x2w_local::BindingFacts buffer_facts;
            buffer_facts.from_table = srv.source == 0;
            buffer_facts.entry_readable = entry_ok;
            buffer_facts.token_magic = entry_ok && words[0] == view_token_magic;
            buffer_facts.token_known = buffer_token_known;
            buffer_facts.bound = bound;
            count_null_binding(x2w_local::classify_null_binding(buffer_facts));  // interlocked counters: no draw_lock needed
        };
        if (placed.reject || (placed.owner && !InterlockedCompareExchange(&placed.owner->own().refs, 0, 0))) {
            buffer_view(dest, nullptr, 0, 0);
            count_buffer(false);
            continue;
        }
        if (placed.owner && bytes % 4 != 0) {
            // Preserve tight byte-sized streams. The desktop raw view is rounded
            // to a DWORD, so initialise its final padding without reading the guest.
            const auto padded = (bytes + 3) & ~std::size_t{3};
            ID3D12Resource* chunk = nullptr;
            auto* target = readable_window(address, bytes) >= bytes ? reserve_upload_ring(padded, &offset, &chunk, nullptr) : nullptr;
            if (target) {
                std::memcpy(target, reinterpret_cast<const void*>(address), bytes);
                std::memset(target + bytes, 0, padded - bytes);
            }
            buffer_view(dest, target ? chunk : nullptr, offset, padded);
            count_buffer(target != nullptr);
            continue;
        }
        if (placed.alias && address % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT == 0 && bytes) {
            require_state(self, placed.alias, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            buffer_view(dest, placed.alias, address - reinterpret_cast<u64>(placed.owner->own().title_memory), bytes);
            continue;
        }
        if (address && srv_in_place_min() && bytes >= srv_in_place_min() && (mesh || srv_in_place_all())) {
            ID3D12Resource* window = nullptr;
            u64 at = 0;
            const bool in_window = desktop_range(address, bytes, &window, &at);
            if (in_window && at % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT == 0) {
                // States: only a window this list already tracks is transitioned (window_read_state). An untracked window
                // gets no barrier, as translated compute binds windows in place (implicit buffer promotion): the indirect
                // stage and compute may have it promoted to UNORDERED_ACCESS, which a COMMON-based barrier would contradict.
                bool tracked = false;
                for (const auto& known : pass.states) tracked |= known.first == window;
                if (tracked) window_read_state(self, window);
                buffer_view(dest, window, at, bytes);
                count_buffer(true);
                srvc_note(mesh ? 2 : 1, srvc_in_place, bytes);
                static LONG shown;
                if (first_calls(shown, 4))
                    log("x2w-draw   %016llx srv %u: 0x%llx (%zu bytes) bound in place at window %p +0x%llx (X2W_DRAW_SRV_IN_PLACE)", tr->key, i,
                        address, bytes, static_cast<void*>(window), at);
                continue;
            }
            InterlockedIncrement64(&srvc_in_place_missed[mesh ? 2 : 1][in_window ? 1 : 0]);
            static LONG missed_shown;
            if (first_calls(missed_shown, 8))
                log("x2w-draw   %016llx srv %u: 0x%llx (%zu bytes) not bound in place: %s; copied as before", tr->key, i, address, bytes,
                    in_window ? "window offset not 16-aligned" : "no single desktop window (desktop_range)");
        }
        const bool gpu_written_range = address && bytes && gpu_written_overlaps(address, bytes);  // its own lock
        if (gpu_written_range) {
            ID3D12Resource* window = nullptr;
            u64 at = 0;
            if (desktop_range(address, bytes, &window, &at) && at % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT == 0) {
                window_read_state(self, window);
                buffer_view(dest, window, at, bytes);
                count_buffer(true);
                InterlockedIncrement(&rt_srv_in_place);
                static LONG shown;
                if (first_calls(shown, 4))
                    log("x2w-draw-rt   %016llx srv %u: 0x%llx (%zu bytes) is written on the GPU (draw uav or copy); bound in place at window %p +0x%llx",
                        tr->key, i, address, bytes, static_cast<void*>(window), at);
                continue;
            }
            InterlockedIncrement(&rt_srv_stale_copy);
        }
        frame_copy_class = srv.source == 0 && entry && words[0] && words[0] != view_token_magic ? 3 : mesh ? 2 : 1;
        ID3D12Resource* chunk = nullptr;
        bool have = false;
        if (address && bytes && readable_window(address, bytes) >= bytes) {
            const std::size_t min_bytes = draw_copy_split_min();
            const bool split_copy = !split && min_bytes && bytes >= min_bytes && !mesh && !indirect && !indirect_pending(self);
            if (split) {
                have = frame_copy_unlocked(address, bytes, &offset, &chunk, 100000 + __LINE__, 100001 + __LINE__);
            } else {
                have = (split_copy && compare_draw_copy(address, bytes, &offset, &chunk)) || frame_copy(address, bytes, &offset, &chunk);
            }
        }
        frame_copy_class = 0;  // thread_local: no lock needed
        // Experiment (X2W_DRAW_NULL_IDENTITY=1): an empty buffer slot reads as repeated identity
        // matrices instead of zeros. The start-up screen's last pass takes its world matrix from a
        // slot the title leaves empty here; why is not known yet.
        static const bool identity_for_null = x2w_local::env_on("X2W_DRAW_NULL_IDENTITY");
        if (!have && formatted.entry) {
            InterlockedIncrement(&formatted_draws_refused);
            static LONG refused;
            if (first_calls(refused, 6))
                log("x2w-draw %016llx srv %u: formatted buffer refused (unresolved buffer)", tr->key, i);
            ReleaseSRWLockExclusive(&draw_lock);
            return false;
        }
        if (!have && identity_for_null) {
            SplitRelock relock(split, 100000 + __LINE__);  // the shared identity array is rewritten here
            static float identity[64];
            for (int k = 0; k < 64; ++k) identity[k] = (k % 16) % 5 == 0 ? 1.0f : 0.0f;
            address = reinterpret_cast<u64>(identity);
            bytes = sizeof(identity);
            have = frame_copy(address, bytes, &offset, &chunk);
        }
        buffer_view(dest, have ? chunk : nullptr, offset, bytes);
        count_buffer(have);
        if (x2w_local::first_calls(tr->logged_bindings, 8) || trace_frame_active) {
            char head[3 * 48 + 1] = "";
            if (have) hex_text(reinterpret_cast<const void*>(address), bytes < 48 ? bytes : 48, head, sizeof(head));
            log("x2w-draw   %016llx srv %u buffer source %d(%d,%d): entry 0x%llx words %08x %08x %08x %08x -> address 0x%llx, %zu bytes, stride %u: %s [%s]",
                tr->key, i, srv.source, srv.a, srv.b, entry, words[0], words[1], words[2], words[3], address, bytes, strides[i],
                have ? "bound" : "NULL", head);
        }
        if (trace_frame_active && have && bytes > 48) {
            for (std::size_t at = 0; at < bytes && at < 512; at += 256) {  // 256 bytes per line (log lines hold 1.5 KiB)
                char part[3 * 256 + 4] = "";
                const std::size_t n = bytes - at < 256 ? bytes - at : 256;
                hex_text(reinterpret_cast<const void*>(address + at), n, part, sizeof(part));
                log("x2w-frame srv-bytes %016llx %u +%zu of %zu: %s", tr->key, i, at, bytes, part);
            }
        }
    }
    static LONG checks;
    if ((InterlockedIncrement(&checks) & 63) == 1) {
        SplitRelock relock(split, 100000 + __LINE__);
        check_device("translated draw views", 0);
    }
    std::uint32_t values[32] = {};
    for (std::size_t i = 0; i < tr->uds.size(); ++i) {
        const Translation::Ud& ud = tr->uds[i];
        values[i] = ud.kind == Translation::rect ? (rect && !indexed_rect ? 1u : 0u)
            : ud.kind == Translation::base_vertex ? static_cast<std::uint32_t>((indexed ? base_vertex : static_cast<INT>(start)) - vertex_first)
            : ud.kind == Translation::start_instance ? start_instance
            : ud.kind == Translation::root_constant ? ((ud.a >= 0 && ud.a < 16 && ud.b >= 0 && ud.b < 8) ? pass.constants[ud.a][ud.b] : 0)
            : ud.kind == Translation::stride && ud.a >= 0 && ud.a < draw_srv_limit ? strides[ud.a]
            : ud.kind == Translation::buf_word3 && ud.a >= 0 && ud.a < draw_srv_limit ? word3[ud.a]
            : ud.kind == Translation::uav_word3 && ud.a >= 0 && ud.a < 8 ? uav_word3[ud.a]
            : ud.kind == Translation::uav_base && ud.a >= 0 && ud.a < 8 ? uav_bases[ud.a]
            : ud.kind == Translation::uav_stride && ud.a >= 0 && ud.a < 8 ? uav_strides[ud.a]
            : ud.kind == Translation::d16_unorm16 && ud.a >= 0 && ud.a < 32 ? d16_unorm16[ud.a]
            : ud.kind == Translation::desc_word3 && ud.a >= 0 && ud.a < draw_srv_limit ? formatted_buffers[ud.a].words[3]
            : ud.kind == Translation::desc_word ? draw_descword_value(pass, ud.a, ud.b, tr->key) : 0;
        if (ud.kind == Translation::buf_word3) {
            InterlockedIncrement(values[i] ? &rt_word3_set : &rt_word3_zero);
            static LONG shown;
            if (first_calls(shown, 1))
                log("x2w-draw-rt bufword3 first use: pipeline %016llx entry value %zu = word 3 of srv %d = 0x%08x", tr->key, i, ud.a, values[i]);
        }
    }
    if (mesh && trace_frame_active) record_trace_mesh_heap(self, *tr);
    if (dumping) {
        SplitRelock relock(split, 100000 + __LINE__);
        char note[600];
        int used = std::snprintf(note, sizeof(note), "entry constants (root parameter 0), kind(a,b) per value:");
        for (std::size_t i = 0; i < tr->uds.size() && used > 0 && used < static_cast<int>(sizeof(note)) - 24; ++i)
            used += std::snprintf(note + used, sizeof(note) - used, " %d(%d,%d)", static_cast<int>(tr->uds[i].kind), tr->uds[i].a, tr->uds[i].b);
        mesh_dump_file("entry.bin", values, tr->uds.size() * sizeof(values[0]), note);
        mesh_dump_file("root_constants.bin", pass.constants, sizeof(pass.constants),
                       "the list's 32-bit root constants as set by the title, [parameter 0-15][word 0-7] u32");
        // The heap buffers a key reads through a root constant (mesh 063500: bit 28 set = the meshlet index comes from
        // a heap buffer whose descriptor index is bits 0-9). Each root constant the translation reads (a `udsrc ..
        // rootconst p w` entry) with bit 28 set is taken as such a reference (ASSUMED; a word that is really something
        // else gives one extra file); other root-constant words are not scanned (floats near 1 have bit 28 set).
        // heap<index>_desc.bin is the 32-byte title descriptor, heap<index>.bin the buffer it names (its readable
        // part, up to 256 MiB).
        u64 heap_address = 0, heap_bytes = 0;
        if (pass.source_view_heap) {
            AcquireSRWLockShared(&note_lock);
            for (const HeapInfo& h : heap_infos())
                if (h.wrapper == pass.source_view_heap && h.title_cpu) heap_address = h.title_cpu, heap_bytes = u64{h.logical_count} * h.increment;
            ReleaseSRWLockShared(&note_lock);
        }
        if (!heap_address) title_view_heap(pass, &heap_address, &heap_bytes);
        std::uint32_t done_index[16];
        unsigned done = 0;
        for (std::size_t u = 0; u < tr->uds.size() && done < 16; ++u) {
                const Translation::Ud& ud = tr->uds[u];
                if (ud.kind != Translation::root_constant || ud.a < 0 || ud.a >= 16 || ud.b < 0 || ud.b >= 8) continue;
                const int p = ud.a, w = ud.b;
                const std::uint32_t c = pass.constants[p][w];
                if (!(c & (1u << 28))) continue;
                const std::uint32_t index = c & 0x3ff;
                bool again = false;
                for (unsigned k = 0; k < done; ++k) again |= done_index[k] == index;
                if (again) continue;
                done_index[done++] = index;
                char file[32], note[300];
                const u64 slot = heap_address + u64{index} * 32;
                const bool slot_ok = heap_address && u64{index} * 32 + 32 <= heap_bytes && readable_window(slot, 32) >= 32;
                std::uint32_t d[8] = {};
                if (slot_ok) std::memcpy(d, reinterpret_cast<const void*>(slot), sizeof(d));
                std::snprintf(file, sizeof(file), "heap%u_desc.bin", index);
                std::snprintf(note, sizeof(note), "title heap descriptor %u (root constant [%d][%d] = 0x%08x) at 0x%llx, heap 0x%llx %llu bytes%s", index,
                              p, w, c, static_cast<unsigned long long>(slot), static_cast<unsigned long long>(heap_address),
                              static_cast<unsigned long long>(heap_bytes), slot_ok ? "" : " (not readable: no bytes)");
                mesh_dump_file(file, slot_ok ? d : nullptr, slot_ok ? sizeof(d) : 0, note);
                if (!slot_ok) continue;
                u64 at = 0;
                std::size_t n = 0;
                UINT stride = 0;
                const char* how = "hardware buffer descriptor";
                if (d[0] == view_token_magic) {  // a view the title created (as the srv table path resolves it)
                    ViewRecord record{};
                    AcquireSRWLockShared(&note_lock);
                    const bool known = d[1] < view_records().size();
                    if (known) record = view_records()[d[1]];
                    ReleaseSRWLockShared(&note_lock);
                    how = known ? "view token" : "view token, unknown view";
                    if (known && record.slot == 17 && record.desc_bytes >= 12) {
                        std::uint32_t size = 0;
                        std::memcpy(&at, record.desc, 8);
                        std::memcpy(&size, record.desc + 8, 4);
                        n = size;
                        how = "view token: constant buffer view";
                    } else if (known && record.slot == 18 && record.desc_bytes >= 40) {
                        D3D12_SHADER_RESOURCE_VIEW_DESC view;
                        std::memcpy(&view, record.desc, 40);
                        Wrapper* resource = as_wrapper(reinterpret_cast<const void*>(record.resource));
                        if (view.ViewDimension == D3D12_SRV_DIMENSION_BUFFER && resource && resource->own().title_memory) {
                            const bool typed_view = !view.Buffer.StructureByteStride && view.Format != DXGI_FORMAT_R32_TYPELESS;
                            const UINT element = view.Buffer.StructureByteStride ? view.Buffer.StructureByteStride
                                                 : typed_view ? x2w_startup::typed_buffer_format_bytes(view.Format) : 4;
                            at = reinterpret_cast<u64>(resource->own().title_memory) + view.Buffer.FirstElement * element;
                            n = std::size_t{view.Buffer.NumElements} * element;
                            stride = element;
                            how = typed_view ? "view token: typed buffer view" : "view token: buffer view";
                        } else {
                            how = "view token: not a buffer view of title memory";
                        }
                    }
                } else {
                    at = d[0] | (static_cast<u64>(d[1] & 0xffff) << 32);
                    stride = (d[1] >> 16) & 0x3fff;
                    n = static_cast<std::size_t>(stride ? stride : 1) * d[2];
                }
                if (n > (256u << 20)) n = 256u << 20;
                if (at && n) n = static_cast<std::size_t>(readable_window(at, n));  // the readable part
                const bool range_ok = at && n;
                std::snprintf(file, sizeof(file), "heap%u.bin", index);
                std::snprintf(note, sizeof(note), "buffer of title heap descriptor %u (%s): words %08x %08x %08x %08x, address 0x%llx stride %u%s", index,
                              how, d[0], d[1], d[2], d[3], static_cast<unsigned long long>(at), stride, range_ok ? "" : " (range not readable: no bytes)");
                mesh_dump_file(file, range_ok ? reinterpret_cast<const void*>(at) : nullptr, range_ok ? n : 0, note);
            }
    }
    // A desktop list with no viewport or scissor draws nothing: cover the first render target unless the title set its own.
    if (pass.target0 && (!pass.viewport_set || !pass.scissor_set)) {
        const D3D12_RESOURCE_DESC target = pass.target0->GetDesc();
        // Only the title's own calls set the flags; an automatic default is regenerated when render target 0 changes size.
        if (!pass.viewport_set && (pass.default_viewport_w != target.Width || pass.default_viewport_h != target.Height)) {
            const D3D12_VIEWPORT full{0.0f, 0.0f, static_cast<float>(target.Width), static_cast<float>(target.Height), 0.0f, 1.0f};
            list->RSSetViewports(1, &full);
            pass.default_viewport_w = target.Width;
            pass.default_viewport_h = target.Height;
        }
        if (!pass.scissor_set && (pass.default_scissor_w != target.Width || pass.default_scissor_h != target.Height)) {
            const D3D12_RECT full{0, 0, static_cast<LONG>(target.Width), static_cast<LONG>(target.Height)};
            list->RSSetScissorRects(1, &full);
            pass.default_scissor_w = target.Width;
            pass.default_scissor_h = target.Height;
        }
    }
    if (split) lock_draw(100000 + __LINE__);
    for (UINT j = 0; j < m; ++j)
        if (uav_textures[j]) require_state(self, uav_textures[j], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12Resource* validated = nullptr;
    if (indirect && indexed) {
        validated = validate_indexed_arguments(self, list, *tr, *indirect, indexed_vertices);
        if (!validated) {
            indirect->set_refusal("translated indexed argument validation could not be recorded",
                                  indirect_refusal_code::indexed_validation_failed);
            ReleaseSRWLockExclusive(&draw_lock);
            return false;
        }
    }
    list->SetPipelineState(pipeline);
    // X2W_DEPTH_STATE: the reference of the decoded stencil test is list state in D3D12. It replaces the one a
    // preceding OMSetStencilRef set, for draws of this pipeline only when a depth view is bound.
    const bool scoped_stencil_ref = bound->depth_from_registers && bound->has_stencil_ref &&
                                    pass_dsv_format(*bound, pass) != DXGI_FORMAT_UNKNOWN &&
                                    bound->stencil_ref != pass.stencil_reference;
    if (scoped_stencil_ref) list->OMSetStencilRef(bound->stencil_ref);
    // Heaps before the root signature: a directly-indexed root signature set with no heap bound is an
    // error on the desktop (#1419).
    ID3D12DescriptorHeap* samplers = tr->samplers.heap ? pass.sampler_heap.Get() : nullptr;
    if (!pass.heap_set || pass.bound_sampler_heap != samplers) {
        x2w_startup::bind_translated_heaps(list, draw_heap, samplers);
        pass.heap_set = true;
        pass.bound_sampler_heap = samplers;
    }
    list->SetGraphicsRootSignature(tr->root);
    list->SetGraphicsRoot32BitConstants(0, static_cast<UINT>(tr->uds.size()), values, 0);
    if (n) list->SetGraphicsRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE{draw_heap->GetGPUDescriptorHandleForHeapStart().ptr + u64{first} * draw_heap_increment});
    if (samplers) list->SetGraphicsRootDescriptorTable(x2w_startup::sampler_root_parameter(n), samplers->GetGPUDescriptorHandleForHeapStart());
    if (m) list->SetGraphicsRootDescriptorTable(tr->uav_parameter, D3D12_GPU_DESCRIPTOR_HANDLE{draw_heap->GetGPUDescriptorHandleForHeapStart().ptr + u64{first + n} * draw_heap_increment});
    if (tr->heap) {
        const UINT h0 = first + n + m;
        const auto cpu = [&](UINT k) { return D3D12_CPU_DESCRIPTOR_HANDLE{draw_heap->GetCPUDescriptorHandleForHeapStart().ptr + SIZE_T{h0 + k} * draw_heap_increment}; };
        const auto gpu = [&](UINT k) { return D3D12_GPU_DESCRIPTOR_HANDLE{draw_heap->GetGPUDescriptorHandleForHeapStart().ptr + u64{h0 + k} * draw_heap_increment}; };
        const u64 vinfo_bytes = vinfo_buffer->GetDesc().Width;
        const u64 vdesc_bytes = tr->heapwords ? vdesc_buffer->GetDesc().Width : 0;
        bounded_raw_srv(heap_view.window, heap_view.offset, heap_view.bytes, cpu(0));
        bounded_raw_srv(vinfo_buffer, 0, vinfo_bytes, cpu(1));
        if (tr->heapwords) bounded_raw_srv(vdesc_buffer, 0, vdesc_bytes, cpu(2));
        list->SetGraphicsRootDescriptorTable(tr->heap_parameter, gpu(0));
        list->SetGraphicsRootDescriptorTable(tr->heap_parameter + 1, gpu(1));
        if (tr->heapwords) list->SetGraphicsRootDescriptorTable(tr->vdesc_parameter, gpu(2));
        if (!tr->heap_views_logged) {
            tr->heap_views_logged = true;  // once per key (draw_lock held)
            static LONG shown;
            if (first_calls(shown, 256))
                log("x2w-draw-rt %016llx: bounded heap views: heapmem %llu bytes (title heap %llu), vinfo %llu, vdesc %llu", tr->key,
                    heap_view.bytes, heap_view.heap_bytes, vinfo_bytes, vdesc_bytes);
        }
        // The region starts at the 8 empty views before id 0 (d3d12_compute.inl fill_views).
        list->SetGraphicsRootDescriptorTable(tr->heap_parameter + 2,
            D3D12_GPU_DESCRIPTOR_HANDLE{draw_heap->GetGPUDescriptorHandleForHeapStart().ptr + u64{draw_heap_size} * draw_heap_increment});
        if (InterlockedIncrement(&rt_heap_draws) == 1)
            log("x2w-draw-rt first heap draw: pipeline %016llx, title heap at desktop 0x%llx, views filled %u", tr->key, heap_view.va, views_filled);
    }
    list->IASetPrimitiveTopology(pass.topology == 5 ? D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP : D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    bool drawn = false;
    gpu_time::mark(self, list, gpu_time::c_draw, tr->key, false);
    if (ring_cap_refusals != ring_refusals_at_start) {
        ring_draw_refused(tr->key);
    } else if (indirect) {  // upstream translated indirect draw (see the head of this function)
        if (validated) {
            list->IASetIndexBuffer(&indirect->indices.view);
            list->ExecuteIndirect(tr->indexed_signature.Get(), 1, validated, 4, validated, 0);
        } else {
            require_state(self, indirect->argument, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
            list->ExecuteIndirect(indirect->signature, 1, indirect->argument, indirect->offset, nullptr, 0);
        }
        // The translated root invalidated the desktop arguments of the tracked guest root.
        if (pass.graphics_root) list->SetGraphicsRootSignature(pass.graphics_root);
        pass.srv_bound = 0;
        drawn = true;
    } else if (indirect_pending(self)) { drawn = indirect_draw_issue(self, list, indexed); } else if (indexed) {
        u64 offset = 0;
        const u64 address = index_address;
        const std::size_t bytes = index_total;
        ID3D12Resource* chunk = nullptr;
        if (indices_ok && frame_copy(address, bytes, &offset, &chunk)) {
            const D3D12_INDEX_BUFFER_VIEW view{chunk->GetGPUVirtualAddress() + offset, static_cast<UINT>(bytes),
                                               index_bytes == 2 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT};
            list->IASetIndexBuffer(&view);
            const auto red_samples = red_producer_draw_before(self, list, "translated-draw", tr->key);
            list->DrawIndexedInstanced(count, instances ? instances : 1, 0, 0, start_instance);
            red_producer_draw_after(self, red_samples);
            drawn = true;
        }
    } else {
        const auto red_samples = red_producer_draw_before(self, list, "translated-draw", tr->key);
        list->DrawInstanced(rect ? count / 3 * 6 : count, instances ? instances : 1, 0, start_instance);
        red_producer_draw_after(self, red_samples);
        drawn = true;
    }
    gpu_time::mark(self, list, gpu_time::c_draw, tr->key, true, !drawn);
    if (drawn && m) {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        list->ResourceBarrier(1, &barrier);
        if (InterlockedIncrement(&rt_uav_draws) == 1)
            log("x2w-draw-rt first draw with uavs: pipeline %016llx, %u uavs, %u srvs", tr->key, m, n);
    }
    if (scoped_stencil_ref) list->OMSetStencilRef(pass.stencil_reference);
    if (drawn && !indirect) draw_dump_after(self, list, tr->key, count);
    ReleaseSRWLockExclusive(&draw_lock);
    if (indirect) return drawn;
    if (trace_frame_active && pass.target0) {
        const D3D12_RESOURCE_DESC target = pass.target0->GetDesc();
        log("x2w-frame translated draw %016llx count %u: target0 %p %llux%u format %d, %u targets", tr->key, count,
            static_cast<void*>(pass.target0), target.Width, target.Height, target.Format, pass.target_count);
    }
    if (x2w_local::first_calls(tr->logged_draws, 10) || trace_frame_active || (count > 100 && x2w_local::first_calls(tr->logged_large, 12))) {
        char indices[3 * 24 + 1] = "";
        if (indexed && pass.ib.BufferLocation) hex_text(reinterpret_cast<const void*>(pass.ib.BufferLocation + u64{start} * 2), 24, indices, sizeof(indices));
        log("x2w-draw %s(%u, %u, start %u, base %d) pipeline %016llx topology 0x%x -> %s; entry [%x %x %x %x %x %x]; ib 0x%llx+%u fmt %d [%s]; "
            "vb 0x%llx %u/%u; targets %u", indexed ? "indexed" : "draw", count, instances, start, base_vertex, tr->key, pass.topology,
            drawn ? "drawn" : "skipped (index data)", values[0], values[1], values[2], values[3], values[4], values[5], pass.ib.BufferLocation,
            pass.ib.SizeInBytes, pass.ib.Format, indices, pass.vb[0].BufferLocation, pass.vb[0].SizeInBytes, pass.vb[0].StrideInBytes,
            pass.target_count);
    }
    if (!drawn) last_draw_why = "skipped (index data)";
    return drawn;
}

bool draw_translated(Wrapper* self, bool indexed, UINT count, UINT instances, UINT start, INT base_vertex, UINT start_instance)
{
    return draw_translated(self, indexed, count, instances, start, base_vertex, start_instance, nullptr);
}

const char* draw_translated_indirect(Wrapper* self, ID3D12CommandSignature* signature, ID3D12Resource* argument, u64 offset,
                                     const IndirectIndexView& indices)
{
    TranslatedIndirectDraw draw{signature, argument, offset, indices};
    if (!draw_translated(self, indices.resource != nullptr, 0, 0, 0, 0, 0, &draw)) {
        static LONG refused;
        if (draw.shader_key) {
            const bool first_pair = indirect_draw_refusal_census.note(draw.shader_key, draw.refusal_code);
            if (first_pair || first_calls(refused, 32))
                log("x2w-render translated ExecuteIndirect Draw refused, shader %016llx: %s", draw.shader_key, draw.refusal);
        }
        return draw.refusal;
    }
    InterlockedIncrement(&translated_draws);
    static LONG reported;
    if (first_calls(reported, 16))
        log("x2w-render translated ExecuteIndirect %s recorded, shader %016llx, offset 0x%llx",
            indices.resource ? "DrawIndexed" : "Draw", draw.shader_key, offset);
    return nullptr;
}

// List slot 32, SetDescriptorHeaps. Track the title's shader-visible sampler
// source while translated draws use their own physical descriptor heaps.
void __stdcall list_set_descriptor_heaps(Wrapper* self, UINT count, void* const* heaps)
{
    x2w_startup::SamplerSource source;
    u64 view_heap = 0;
    if (count && count <= 4 && x2w_local::readable(heaps, std::size_t{count} * sizeof(*heaps))) {
        AcquireSRWLockShared(&note_lock);
        for (UINT i = 0; i < count; ++i) {
            for (const HeapInfo& heap : heap_infos()) {
                if (heap.wrapper == reinterpret_cast<u64>(heaps[i]) &&
                    heap.type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV &&
                    heap.logical_count > heap.count)
                    view_heap = heap.wrapper;
                if (heap.wrapper == reinterpret_cast<u64>(heaps[i]) &&
                    heap.type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER && heap.desktop_gpu)
                    source = {static_cast<std::uintptr_t>(heap.title_sampler_cpu), heap.count};
            }
        }
        ReleaseSRWLockShared(&note_lock);
    }
    auto& pass = pass_of(self);
    if (pass.source_view_heap != view_heap)
    {
        for (auto& table : pass.view_tables) table = {};
        pass.compute_tables.clear();
    }
    pass.source_view_heap = view_heap;
    pass.title_samplers = source;
}

// The table at a title GPU handle, looked up in the title's view heap (the one set by slot 32 when known).
x2w_startup::ViewTable resolve_title_table(const Pass& pass, u64 gpu_handle)
{
    x2w_startup::ViewTable table;
    AcquireSRWLockShared(&note_lock);
    for (const HeapInfo& heap : heap_infos()) {
        if (heap.type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) continue;
        if (pass.source_view_heap && heap.wrapper != pass.source_view_heap) continue;
        if (!pass.source_view_heap && heap.logical_count > heap.count) continue;
        table = x2w_startup::resolve_view_table(heap.desktop_gpu, heap.title_cpu, heap.logical_count, heap.increment, gpu_handle);
        if (table.address) break;
    }
    ReleaseSRWLockShared(&note_lock);
    return table;
}

// List slot 36 (desktop 32 + 4), SetGraphicsRootDescriptorTable(parameter, GPU handle): the table's
// address in the title's descriptor memory is kept for the draw.
void __stdcall list_table(Wrapper* self, UINT parameter, u64 gpu_handle)
{
    if (parameter >= 16) return;
    auto& pass = pass_of(self);
    pass.view_tables[parameter] = resolve_title_table(pass, gpu_handle);
}
// List slot 35 (desktop 31 + 4), SetComputeRootDescriptorTable(parameter, GPU handle). Recorded only for a later
// Dispatch; slot 33 keeps the desktop compute root bind.
void __stdcall list_compute_table_desktop(Wrapper* self, UINT parameter, u64 gpu_handle)
{
    static LONG dropped;
    if (!self->own().list_open) return;
    auto& pass = pass_of(self);
    if (!pass.compute_tables.set_table(parameter, resolve_title_table(pass, gpu_handle)) && first_calls(dropped, 4))
        log("x2w-render compute descriptor table dropped: parameter %u (%u parameters are kept)", parameter,
            x2w_startup::ComputeTables::max_parameters);
}
// List slot 40 (desktop 36 + 4), SetGraphicsRoot32BitConstants(parameter, count, data, offset).
void __stdcall list_root_constants(Wrapper* self, UINT parameter, UINT count, const std::uint32_t* data, UINT offset)
{
    static LONG dropped;
    if (count && (parameter >= 16 || offset >= 8 || count > 8 - offset) && first_calls(dropped, 4))
        log("x2w-render root constants %s: parameter %u, offset %u, count %u (16 parameters of 8 words are kept)",
            parameter >= 16 || offset >= 8 ? "dropped" : "truncated", parameter, offset, count);
    if (parameter >= 16 || offset >= 8 || !count || !x2w_local::readable(data, std::size_t{count} * 4)) return;
    Pass& pass = pass_of(self);
    for (UINT i = 0; i < count && offset + i < 8; ++i) pass.constants[parameter][offset + i] = data[i];
    if (pass.root && parameter < pass.root->size() &&
        (*pass.root)[parameter].type == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS &&
        count <= 8 - offset && offset + count <= (*pass.root)[parameter].constants_count)
        pass.constant_words_bound[parameter] |= static_cast<unsigned char>(((1u << count) - 1) << offset);
}
void __stdcall list_root_uav(Wrapper* self, UINT parameter, u64 address)
{
    if (parameter < 16) pass_of(self).root_address[parameter] = address;
}
// List slot 47 (desktop 43 + 4), IASetIndexBuffer(view).
void __stdcall list_index_buffer(Wrapper* self, const D3D12_INDEX_BUFFER_VIEW* view)
{
    pass_of(self).ib = view && x2w_local::readable(view, sizeof(*view)) ? *view : D3D12_INDEX_BUFFER_VIEW{};
}
// List slot 48 (desktop 44 + 4), IASetVertexBuffers(start, count, views).
void __stdcall list_vertex_buffers(Wrapper* self, UINT start, UINT count, const D3D12_VERTEX_BUFFER_VIEW* views)
{
    if (!views || !x2w_local::readable(views, std::size_t{count} * sizeof(*views))) return;
    if (start >= 8) return;
    Pass& pass = pass_of(self);
    for (UINT i = 0; i < count && i < 8 - start; ++i) {
        pass.vb[start + i] = views[i];
        pass.placed_vb[start + i] = placed_buffer_range(self->own().list_device, views[i].BufferLocation, views[i].SizeInBytes);
    }
}
// List slot 13, DrawIndexedInstanced(index count, instances, start index, base vertex, start instance).
void __stdcall list_draw_indexed(Wrapper* self, UINT count, UINT instances, UINT start, INT base_vertex, UINT start_instance)
{
    const bool done = draw_translated(self, true, count, instances, start, base_vertex, start_instance);
    InterlockedIncrement(done ? &translated_draws : &untranslated_draws);
    if (trace_frame_active) {
        const Pass& pass = pass_of(self);
        note_traced_target(self, pass.target0, done);
        const D3D12_RESOURCE_DESC rd = pass.target0 ? pass.target0->GetDesc() : D3D12_RESOURCE_DESC{};
        log("x2w-frame draw-indexed %u %s target %p %llux%u fmt %d key %016llx %s", count, done ? "drawn" : "NOT drawn",
            static_cast<void*>(pass.target0), static_cast<unsigned long long>(rd.Width), rd.Height, rd.Format, last_draw_key,
            done ? "" : (last_draw_why ? last_draw_why : "draw failed"));
    }
}

void __stdcall list_copy_buffer(Wrapper* self, Wrapper* dest, u64 dest_offset, Wrapper* source, u64 source_offset, u64 bytes);

void copy_watch_note(const char* path, u64 to, u64 from, u64 bytes, const char* outcome)
{
    static u64 watched[4];
    static int count = [] {
        char text[128]{};
        int n = 0;
        if (GetEnvironmentVariableA("X2W_D3D12_COPY_WATCH", text, sizeof(text))) {
            for (char* at = text; *at && n < 4;) {
                char* end = nullptr;
                const u64 value = std::strtoull(at, &end, 16);
                if (end == at) break;
                if (value) watched[n++] = value;
                at = *end == ',' ? end + 1 : end;
            }
            log("x2w-copy X2W_D3D12_COPY_WATCH: %d address(es) watched (%s)", n, text);
        }
        return n;
    }();
    static LONG lines[4];
    for (int i = 0; i < count; ++i)
        if (to && bytes && to < watched[i] + 0x100 && watched[i] < to + bytes && first_calls(lines[i], 64)) {
            unsigned char head[32]{};
            char text[3 * 32 + 1] = "";
            if (from && x2w_local::readable(reinterpret_cast<const void*>(from), bytes < 32 ? bytes : 32)) {
                std::memcpy(head, reinterpret_cast<const void*>(from), bytes < 32 ? bytes : 32);
                hex_text(head, bytes < 32 ? bytes : 32, text, sizeof(text));
            }
            log("x2w-copy watch 0x%llx: %s copy to 0x%llx from 0x%llx bytes 0x%llx frame %ld: %s; source [%s]", watched[i], path, to, from,
                bytes, frame_number, outcome, text);
        }
}
bool copy_title_bytes(unsigned char* dest, const unsigned char* source, std::size_t bytes) noexcept;  // d3d12_copy_buffer.inl
void note_issued_dispatch(ID3D12GraphicsCommandList* list, UINT x, UINT y, UINT z, bool indirect);  // d3d12_xs_device.cpp (DRED report)
extern thread_local u64 cx_issue_key;
extern thread_local char cx_issue_path;
constexpr char copy_gpu_source[] = R"(
RWByteAddressBuffer from : register(u0);
RWByteAddressBuffer to : register(u1);
cbuffer c : register(b0) { uint from_at; uint to_at; uint words; uint step; };
[numthreads(64, 1, 1)] void cs(uint3 id : SV_DispatchThreadID)
{
    for (uint i = id.x; i < words; i += step) to.Store(to_at + i * 4, from.Load(from_at + i * 4));
}
)";
ID3D12RootSignature* copy_gpu_root;
ID3D12PipelineState* copy_gpu_pso;
bool ensure_copy_gpu()
{
    static SRWLOCK lock = SRWLOCK_INIT;
    static bool failed;
    if (copy_gpu_pso) return true;
    AcquireSRWLockExclusive(&lock);
    if (!copy_gpu_pso && !failed) {
        ID3D12Device* device = device_of(the_device);
        ID3DBlob *code = nullptr, *errors = nullptr, *serialized = nullptr;
        HRESULT hr = D3DCompile(copy_gpu_source, sizeof(copy_gpu_source) - 1, "copy_gpu", nullptr, nullptr, "cs", "cs_5_0",
                                D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
        if (FAILED(hr) && errors) log("x2w-copy-gpu compile: %.400s", static_cast<const char*>(errors->GetBufferPointer()));
        D3D12_ROOT_PARAMETER params[3]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[0].Descriptor.ShaderRegister = 0;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[1].Descriptor.ShaderRegister = 1;
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[2].Constants.Num32BitValues = 4;
        for (auto& p : params) p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        const D3D12_ROOT_SIGNATURE_DESC root_desc{3, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        if (SUCCEEDED(hr)) hr = D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, nullptr);
        ID3D12RootSignature* root = nullptr;
        if (SUCCEEDED(hr)) hr = device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&root));
        ID3D12PipelineState* pso = nullptr;
        if (SUCCEEDED(hr)) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
            desc.pRootSignature = root;
            desc.CS = {code->GetBufferPointer(), code->GetBufferSize()};
            hr = device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso));
        }
        if (code) code->Release();
        if (errors) errors->Release();
        if (serialized) serialized->Release();
        if (SUCCEEDED(hr)) {
            copy_gpu_root = root;
            copy_gpu_pso = pso;
        } else {
            if (root) root->Release();
            failed = true;
        }
        log("x2w-copy-gpu copy pipeline: 0x%08lx", static_cast<unsigned long>(hr));
    }
    ReleaseSRWLockExclusive(&lock);
    return copy_gpu_pso != nullptr;
}

void __stdcall list_copy_buffer_gpu(Wrapper* self, Wrapper* dest, u64 dest_offset, Wrapper* source, u64 source_offset, u64 bytes)
{
    auto* const to = dest && dest->own().kind == resource_kind ? static_cast<unsigned char*>(dest->own().title_memory) : nullptr;
    auto* const from = source && source->own().kind == resource_kind ? static_cast<const unsigned char*>(source->own().title_memory) : nullptr;
    ID3D12GraphicsCommandList* list = recording(self);
    const u64 to_address = to ? reinterpret_cast<u64>(to) + dest_offset : 0;
    const u64 from_address = from ? reinterpret_cast<u64>(from) + source_offset : 0;
    ID3D12Resource *to_window = nullptr, *from_window = nullptr;
    u64 to_at = 0, from_at = 0;
    int reason = -1;
    if (!list) reason = cf_no_list;
    else if (!to || !from) reason = cf_no_title_memory;
    else if (!bytes || bytes > (u64{256} << 20) || dest_offset > (u64{1} << 40) || source_offset > (u64{1} << 40)) reason = cf_range;
    else if ((to_address | from_address | bytes) & 3) reason = cf_unaligned;
    else if (!desktop_range(to_address, bytes, &to_window, &to_at)) reason = cf_dest_unresolved;
    else if (!desktop_range(from_address, bytes, &from_window, &from_at)) reason = cf_source_unresolved;
    else if (to_window == from_window && to_at < from_at + bytes && from_at < to_at + bytes) reason = cf_overlap;
    const bool copy_list = list && list->GetType() == D3D12_COMMAND_LIST_TYPE_COPY;
    if (reason < 0 && copy_list && to_window == from_window) reason = cf_copy_list_same;
    else if (reason < 0 && !copy_list && !ensure_copy_gpu()) reason = cf_no_pipeline;
    static LONG shown;
    if (reason >= 0) {
        InterlockedIncrement(&cg_fallback[reason]);
        if (first_calls(shown, 16))
            log("x2w-copy-gpu CopyBufferRegion title %p <- %p bytes 0x%llx: record-time fallback: %s", static_cast<void*>(to ? to + dest_offset : nullptr),
                static_cast<const void*>(from ? from + source_offset : nullptr), bytes, copy_fallback_names[reason]);
        list_copy_buffer(self, dest, dest_offset, source, source_offset, bytes);
        return;
    }
    // A source the GPU never wrote (CPU-filled upload memory) is also copied at record time, so the paths that read
    // the destination's title memory when the title records (vertex/index ring copies, misaligned srvs, texture
    // uploads) see what X2W_D3D12_COPY_BUFFER=1 gave them; the GPU copy then writes the same bytes in list order.
    lock_draw(100000 + __LINE__);
    const bool source_gpu_written = gpu_written_overlaps(from_address, bytes);
    ReleaseSRWLockExclusive(&draw_lock);
    copy_watch_note("gpu", to_address, from_address, bytes, source_gpu_written ? "GPU copy (source GPU-written)" : "GPU copy + record-time copy");
    if (!source_gpu_written && x2w_local::readable(from + source_offset, static_cast<std::size_t>(bytes)) &&
        copy_title_bytes(to + dest_offset, from + source_offset, static_cast<std::size_t>(bytes)))
        InterlockedIncrement(&cg_record_too);
    if (copy_list) {
        // A copy-type list holds no compute or draw work: windows there are only ever moved by these copies, so the
        // tracked states are exact (back to COMMON at Close).
        require_state(self, from_window, D3D12_RESOURCE_STATE_COPY_SOURCE);
        require_state(self, to_window, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(to_window, to_at, from_window, from_at, bytes);
        InterlockedIncrement(&cg_copy_list);
    } else {
        // Both windows in UNORDERED_ACCESS in this list's tracked states (from whatever this list's draws and copies
        // left them in; moved back to COMMON at Close).
        require_state(self, from_window, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        require_state(self, to_window, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        // Root UAVs at 256-byte steps inside the windows; the remainder goes to the shader as byte offsets.
        const u64 to_base = to_at & ~u64{255}, from_base = from_at & ~u64{255};
        const std::uint32_t words = static_cast<std::uint32_t>(bytes / 4);
        const UINT groups = words / 64 + 1 > 4096 ? 4096 : words / 64 + 1;
        const std::uint32_t values[4] = {static_cast<std::uint32_t>(from_at - from_base), static_cast<std::uint32_t>(to_at - to_base), words, groups * 64};
        D3D12_RESOURCE_BARRIER uav{};
        uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        list->ResourceBarrier(1, &uav);
        list->SetPipelineState(copy_gpu_pso);
        list->SetComputeRootSignature(copy_gpu_root);
        list->SetComputeRootUnorderedAccessView(0, from_window->GetGPUVirtualAddress() + from_base);
        list->SetComputeRootUnorderedAccessView(1, to_window->GetGPUVirtualAddress() + to_base);
        list->SetComputeRoot32BitConstants(2, 4, values, 0);
        // The copy pass is a Dispatch op in DRED too; record it (path 'b', key 0) so the report's Dispatch op count matches.
        const u64 issue_key = cx_issue_key;
        const char issue_path = cx_issue_path;
        cx_issue_key = 0;
        cx_issue_path = 'b';
        note_issued_dispatch(list, groups, 1, 1, false);
        cx_issue_key = issue_key;
        cx_issue_path = issue_path;
        gpu_time::mark(self, list, gpu_time::c_copy, bytes, false);
        list->Dispatch(groups, 1, 1);
        gpu_time::mark(self, list, gpu_time::c_copy, bytes, true);
        list->ResourceBarrier(1, &uav);
        // Graphics work after this point: a title pipeline with a desktop pipeline expects its state back (translated
        // draws and compute set theirs on every call).
        const Pass& pass = pass_of(self);
        if (pass.pipeline_real && pass.pipeline) list->SetPipelineState(static_cast<ID3D12PipelineState*>(pass.pipeline->own().real));
    }
    lock_draw(100000 + __LINE__);
    note_gpu_written(to_address, bytes);
    report_draw_rt();
    ReleaseSRWLockExclusive(&draw_lock);
    InterlockedIncrement(&cg_gpu);
    if (to_window == from_window) InterlockedIncrement(&cg_same_window);
    InterlockedAdd64(&cg_bytes, static_cast<LONG64>(bytes));
    if (first_calls(shown, 16))
        log("x2w-copy-gpu CopyBufferRegion title %p <- %p bytes 0x%llx: on the GPU, window %p +0x%llx <- window %p +0x%llx",
            static_cast<void*>(to + dest_offset), static_cast<const void*>(from + source_offset), bytes, static_cast<void*>(to_window), to_at,
            static_cast<void*>(from_window), from_at);
}
