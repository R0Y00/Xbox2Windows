// Pipeline creation records shader metadata and owns a copy of graphics stream state. First use requests a host
// root signature and PSO. Pipelines with matching stages and state share the result.
//
// Render-target formats use desktop_format with packed_target_format stand-ins as required by the adapter. Depth
// and input elements use desktop_format. A draw with different bound targets requests a variant using
// GpuTargets.
//
// X2W_GPU_MESH=0 disables mesh pipelines. X2W_GPU_ROOT_SRV_DIRECT=1 uses unbounded root descriptors for
// title SRVs and UAVs in place of bounded one-view tables.

// One title pipeline: what it is made of, and the host pipelines made for it.
using GpuJobs = x2w_gpu::VpBuildQueue<x2w_gpu::VpPending>;

// Park the compiler worker in the exception callback so driver frames and driver-owned locks stay intact. The
// monitor publishes the terminal result and starts a replacement. Slots live for the process.
struct GpuCompileFault {
    GpuJobs::ActiveBuild* owner = nullptr;
    ULONG_PTR address = 0, instruction = 0;
    DWORD tid = 0;
    std::atomic<bool> ready{false};
};
constexpr unsigned gpu_fault_limit = 8;
GpuCompileFault gpu_faults[gpu_fault_limit];
std::atomic<unsigned> gpu_fault_count{0};
HANDLE gpu_fault_event = nullptr;
uintptr_t gpu_compiler_base = 0, gpu_compiler_end = 0;
thread_local bool gpu_in_compiler = false;

std::filesystem::path gpu_cache_setting(const wchar_t* name)
{
    const DWORD size = GetEnvironmentVariableW(name, nullptr, 0);
    if (!size) return {};
    std::wstring path(size, L'\0');
    const DWORD chars = GetEnvironmentVariableW(name, path.data(), size);
    if (!chars || chars >= size) return {};
    path.resize(chars);
    return path;
}

// The copied reference list stays read-only. The local list belongs to the
// selected adapter/driver and records exact identities from this runtime.
const std::filesystem::path& gpu_local_fault_cache()
{
    static const auto path = gpu_cache_setting(L"X2W_GPU_LOCAL_FAULT_FILE");
    return path;
}

LONG CALLBACK gpu_compiler_fault(EXCEPTION_POINTERS* info)
{
    auto* owner = GpuJobs::active;
    const auto& fault = *info->ExceptionRecord;
    const auto ip = reinterpret_cast<uintptr_t>(fault.ExceptionAddress);
    if (!owner || !gpu_in_compiler || fault.ExceptionCode != EXCEPTION_ACCESS_VIOLATION ||
        fault.NumberParameters < 2 || fault.ExceptionInformation[0] > 1 || fault.ExceptionInformation[1] >= 0x10000 ||
        ip < gpu_compiler_base || ip >= gpu_compiler_end) return EXCEPTION_CONTINUE_SEARCH;
    const auto index = gpu_fault_count.fetch_add(1, std::memory_order_relaxed);
    if (index >= gpu_fault_limit) return EXCEPTION_CONTINUE_SEARCH;
    gpu_in_compiler = false;
    auto& slot = gpu_faults[index];
    slot.owner = owner;
    slot.address = fault.ExceptionInformation[1];
    slot.instruction = ip;
    slot.tid = GetCurrentThreadId();
    slot.ready.store(true, std::memory_order_release);
    SetEvent(gpu_fault_event);
    for (;;) Sleep(INFINITE);
}

void gpu_watch_compiler_faults()
{
    while (WaitForSingleObject(gpu_fault_event, INFINITE) == WAIT_OBJECT_0) {
        for (auto& slot : gpu_faults) {
            if (!slot.ready.exchange(false, std::memory_order_acquire)) continue;
            const auto& pending = *slot.owner->job->state;
            std::string keys;
            for (const auto& shader : pending.shaders) {
                const auto* record = static_cast<const x2w_gpu::GpuShader*>(shader.tag);
                keys += " stage " + std::to_string(shader.stage) + "=" + (record ? record->key : "missing");
                if (record && !pending.graphics && pending.shaders.size() == 1 && shader.stage == x2w_gpu::vp_stage_compute) {
                    const auto id = x2w_gpu::gpu_compute_pending_key(record->key);
                    keys += " warm-bad=" + id;
                    if (!gpu_local_fault_cache().empty()) {
                        std::ofstream out(gpu_local_fault_cache(), std::ios::app);
                        if (!(out << id << " compiler fault on compute " << record->key << '\n').flush())
                            log("x2w-gpu could not save local compiler fault: %s", id.c_str());
                    }
                }
            }
            log("x2w-gpu compiler worker parked: tid %lu amdxc64+%llx access %llx%s", slot.tid,
                static_cast<unsigned long long>(slot.instruction - gpu_compiler_base),
                static_cast<unsigned long long>(slot.address), keys.c_str());
            try { slot.owner->lost(std::make_exception_ptr(std::runtime_error("AMD compiler worker parked:" + keys))); }
            catch (const std::system_error& e) { log("x2w-gpu could not replace compiler worker: %s", e.what()); }
        }
    }
}

void gpu_install_compiler_guard()
{
    if (x2w_gpu::gpu_env_is("X2W_GPU_COMPILER_GUARD", '0')) return;
    HMODULE compiler = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN, L"amdxc64.dll", &compiler)) return;
    gpu_compiler_base = reinterpret_cast<uintptr_t>(compiler);
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(compiler);
    const auto* pe = reinterpret_cast<const IMAGE_NT_HEADERS*>(gpu_compiler_base + dos->e_lfanew);
    gpu_compiler_end = gpu_compiler_base + pe->OptionalHeader.SizeOfImage;
    gpu_fault_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!gpu_fault_event) return;
    const auto handler = AddVectoredExceptionHandler(1, gpu_compiler_fault);
    if (!handler) {
        CloseHandle(gpu_fault_event);
        gpu_fault_event = nullptr;
        return;
    }
    try { std::thread(gpu_watch_compiler_faults).detach(); }
    catch (const std::system_error& e) {
        RemoveVectoredExceptionHandler(handler);
        CloseHandle(gpu_fault_event);
        gpu_fault_event = nullptr;
        log("x2w-gpu compiler guard unavailable: %s", e.what());
    }
}
struct GpuRequest {
    std::shared_ptr<x2w_gpu::VpPending> pending = std::make_shared<x2w_gpu::VpPending>();
    std::once_flag requested;
    GpuJobs::Handle job;
    std::atomic<bool> reported{false};
    std::atomic<bool> timeout_reported{false};
};
namespace vbind {
bool compute_capture_on()
{
    static const bool enabled = x2w_local::env_on("X2W_TRACE_COMPUTE_TEXTURES");
    return enabled;
}
bool pixel_capture_on()
{
    static const bool enabled = x2w_local::env_on("X2W_TRACE_PIXEL_TEXTURES");
    return enabled;
}
}

const x2w_gpu::ambient::Settings& indirect_light_settings()
{
    static const auto settings = [] {
        x2w_gpu::ambient::Settings out{};
        auto positive = [](const char* key) {
            char value[48]{};
            const DWORD n = GetEnvironmentVariableA(key, value, sizeof(value));
            char* end = nullptr;
            if (!n || n >= sizeof(value)) return 0.0f;
            const float f = std::strtof(value, &end);
            return end && !*end && std::isfinite(f) && f > 0 ? f : 0.0f;
        };
        out.level = positive("X2W_GPU_INDIRECT_LIGHT");
        out.air = positive("X2W_GPU_INDIRECT_LIGHT_AIR");
        return out;
    }();
    return settings;
}

struct GpuPipelineEntry {
    x2w_gpu::ambient::Role light_role = x2w_gpu::ambient::Role::none;
    GpuRequest base;
    bool listed_bad = false;
    bool exposure_reset = false;
    std::vector<x2w_gpu::GpuStage> stages;  // per pending.shaders entry
    std::vector<const x2w_gpu::GpuShader*> records;
    std::unique_ptr<std::string> missing_keys; // requested keys absent from the shader DB, only when tracing drops
    // The pipeline for the stream's own targets, then variants for bound targets that differ.
    std::mutex lock;
    std::vector<std::pair<x2w_gpu::GpuTargets, std::unique_ptr<GpuRequest>>> variants;
};

// A made host pipeline and its view for WP3/WP4, never freed (draws recorded with it may still be in flight).
struct GpuMade {
    std::shared_ptr<const x2w_gpu::VpHostPipeline> host;
    x2w_gpu::GpuPipeline view;
};

x2w_gpu::VpPipelineCache& gpu_pipeline_cache()
{
    static auto* cache = new x2w_gpu::VpPipelineCache;
    return *cache;
}

// The views by host pipeline (one per host pipeline, shared like it).
std::mutex gpu_made_lock;
std::map<const x2w_gpu::VpHostPipeline*, GpuMade*>& gpu_made()
{
    static auto* made = new std::map<const x2w_gpu::VpHostPipeline*, GpuMade*>;
    return *made;
}

std::uint32_t gpu_vp_stage(x2w_gpu::GpuStage s)
{
    using x2w_gpu::GpuStage;
    switch (s) {
    case GpuStage::vertex: return x2w_gpu::vp_stage_vertex;
    case GpuStage::pixel: return x2w_gpu::vp_stage_pixel;
    case GpuStage::domain: return x2w_gpu::vp_stage_domain;
    case GpuStage::hull: return x2w_gpu::vp_stage_hull;
    case GpuStage::geometry: return x2w_gpu::vp_stage_geometry;
    case GpuStage::compute: return x2w_gpu::vp_stage_compute;
    case GpuStage::amplification: return x2w_gpu::vp_stage_amplification;
    case GpuStage::mesh: return x2w_gpu::vp_stage_mesh;
    default: return x2w_gpu::vp_stage_none;
    }
}

x2w_gpu::GpuStage gpu_stage_of(std::uint32_t vp)
{
    using x2w_gpu::GpuStage;
    switch (vp) {
    case x2w_gpu::vp_stage_vertex: return GpuStage::vertex;
    case x2w_gpu::vp_stage_pixel: return GpuStage::pixel;
    case x2w_gpu::vp_stage_domain: return GpuStage::domain;
    case x2w_gpu::vp_stage_hull: return GpuStage::hull;
    case x2w_gpu::vp_stage_geometry: return GpuStage::geometry;
    case x2w_gpu::vp_stage_compute: return GpuStage::compute;
    case x2w_gpu::vp_stage_amplification: return GpuStage::amplification;
    case x2w_gpu::vp_stage_mesh: return GpuStage::mesh;
    default: return GpuStage::count;
    }
}

// D3D12_FORMAT_SUPPORT1 of a format on the device, asked once per format.
UINT gpu_format_support1(DXGI_FORMAT format)
{
    static std::atomic<UINT> known[256];  // support | 0x80000000 once asked
    const UINT f = static_cast<UINT>(format);
    if (f < 256) {
        const UINT seen = known[f].load(std::memory_order_relaxed);
        if (seen) return seen & 0x7fffffffu;
    }
    UINT support = 0;
    if (the_device) {
        D3D12_FEATURE_DATA_FORMAT_SUPPORT data{format};
        if (SUCCEEDED(device_of(the_device)->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &data, sizeof(data))))
            support = static_cast<UINT>(data.Support1) & 0x7fffffffu;
    }
    if (f < 256) known[f].store(support | 0x80000000u, std::memory_order_relaxed);
    return support;
}

// Render target formats as X2W makes such textures: desktop_format, then the packed stand-in when the adapter cannot
// render to the format and can to the stand-in (packed_target_stand_in's rule for a render target).
DXGI_FORMAT gpu_target_format(std::uint32_t console)
{
    const DXGI_FORMAT format = desktop_format(static_cast<DXGI_FORMAT>(console));
    const auto* mapping = x2w_startup::packed_target_mapping(format);
    if (!mapping) return format;
    const UINT target = D3D12_FORMAT_SUPPORT1_RENDER_TARGET;
    if (gpu_format_support1(format) & target) return format;
    return gpu_format_support1(mapping->replacement) & target ? mapping->replacement : format;
}

DXGI_FORMAT gpu_plain_format(std::uint32_t console) { return desktop_format(static_cast<DXGI_FORMAT>(console)); }

bool gpu_readable(const void* p, std::size_t bytes) { return x2w_local::readable(p, bytes); }

const std::wstring& gpu_reference_library_path()
{
    static const std::wstring path = [] {
        constexpr auto name = L"X2W_GPU_REFERENCE_LIBRARY";
        const DWORD size = GetEnvironmentVariableW(name, nullptr, 0);
        std::wstring value(size, L'\0');
        if (!size) return value;
        const DWORD chars = GetEnvironmentVariableW(name, value.data(), size);
        value.resize(chars < size ? chars : 0);
        return value;
    }();
    return path;
}

HRESULT gpu_load_reference_compute(ID3D12Device* device, const std::vector<unsigned char>& root,
                                       const D3D12_COMPUTE_PIPELINE_STATE_DESC& desc, ID3D12PipelineState** out)
{
    struct Entry {
        Microsoft::WRL::ComPtr<ID3D12Device> device;
        std::once_flag once;
        std::unique_ptr<x2w_gpu::VpReferenceLibrary> library;
        HRESULT opened = E_FAIL;
    };
    static std::mutex lock;
    static auto* entries = new std::map<ID3D12Device*, std::unique_ptr<Entry>>;
    Entry* entry;
    {
        std::lock_guard<std::mutex> held(lock);
        auto& slot = (*entries)[device];
        if (!slot) {
            slot = std::make_unique<Entry>();
            slot->device = device;
        }
        entry = slot.get();
    }
    std::call_once(entry->once, [&] {
        auto library = std::make_unique<x2w_gpu::VpReferenceLibrary>();
        entry->opened = library->open(device, gpu_reference_library_path().c_str());
        if (SUCCEEDED(entry->opened)) entry->library = std::move(library);
        log("x2w-gpu read-only reference pipeline library: open 0x%08lx", static_cast<unsigned long>(entry->opened));
    });
    if (FAILED(entry->opened)) return entry->opened;
    const auto name = x2w_gpu::vp_compute_library_name(root, desc.CS);
    const auto start = GetTickCount64();
    const HRESULT hr = entry->library->load(name, desc, out);
    static LONG hits, misses;
    if (first_calls(SUCCEEDED(hr) ? hits : misses, 24) || GetTickCount64() - start >= 3000)
        log("x2w-gpu reference compute %ls: load 0x%08lx in %llu ms", name.c_str(), static_cast<unsigned long>(hr),
            GetTickCount64() - start);
    return hr;
}

x2w_gpu::VpMakeOptions gpu_make_options()
{
    static const x2w_gpu::VpMakeOptions options = [] {
        x2w_gpu::VpMakeOptions o;
        o.allow_mesh = x2w_gpu::gpu_mesh_on();
        if (!gpu_reference_library_path().empty()) o.load_compute = gpu_load_reference_compute;
        o.root.bounded_root_views = !x2w_gpu::gpu_env_is("X2W_GPU_ROOT_SRV_DIRECT", '1');
        return o;
    }();
    return options;
}

const x2w_gpu::VpDeviceCaps& gpu_caps()
{
    static const x2w_gpu::VpDeviceCaps caps = [] {
        const auto c = x2w_gpu::vp_device_caps(device_of(the_device));
        log("x2w-gpu pipelines: mesh shaders %d, depth bounds %d, aniso point-mip %d, root views %s, mesh %s", c.mesh_shaders,
            c.depth_bounds, c.aniso_point_mip, gpu_make_options().root.bounded_root_views ? "one-view tables" : "root descriptors",
            gpu_make_options().allow_mesh ? "on" : "off (X2W_GPU_MESH=0)");
        return c;
    }();
    return caps;
}

const x2w_gpu::GpuWarmBad& gpu_warm_bad()
{
    static const auto list = [] {
        x2w_gpu::GpuWarmBad result;
        const auto reference = gpu_cache_setting(L"X2W_GPU_WARM_BAD_FILE");
        for (const auto& path : {reference, gpu_local_fault_cache()}) {
            if (path.empty()) continue;
            std::ifstream input(path);
            if (!input) {
                if (path == reference) log("x2w-gpu could not open the configured warm-bad cache");
                continue;
            }
            x2w_gpu::GpuWarmBad parsed;
            parsed.read(input);
            if (input.bad()) log("x2w-gpu failed to read a configured warm-bad cache");
            else result.keys.merge(parsed.keys);
        }
        if (!reference.empty() || !gpu_local_fault_cache().empty())
            log("x2w-gpu loaded %zu warm-bad pipeline keys (single-stage compute matching)", result.keys.size());
        return result;
    }();
    return list;
}

bool gpu_drop_trace()
{
    static const bool on = x2w_local::env_on("X2W_GPU_DROP_TRACE");
    return on;
}

void gpu_queue_request(GpuRequest& request, bool initialization);
bool gpu_synchronous_build();

bool gpu_create_pipeline(Wrapper* pipeline, const GpuCreateDesc& desc)
{
    if (!pipeline || desc.unreadable || !desc.stage_count) return false;
    auto entry = std::make_unique<GpuPipelineEntry>();
    entry->base.pending->creation_slot = static_cast<std::uint32_t>(desc.slot);
    for (std::uint32_t i = 0; i < desc.stage_count && i < std::size(desc.stages); ++i) {
        const auto& code = desc.stages[i];
        const x2w_gpu::GpuShader* record = x2w_gpu::gpu_shader_by_key(code.key, code.stage);
        entry->base.pending->add(x2w_gpu::vp_record_view(record, gpu_vp_stage(code.stage)));
        entry->stages.push_back(code.stage);
        entry->records.push_back(record);
        static const char* const exposure_key = std::getenv("X2W_EXPOSURE_RESET_KEY");
        if (code.stage == x2w_gpu::GpuStage::pixel && exposure_key && std::strcmp(code.key, exposure_key) == 0)
            entry->exposure_reset = true;
        if (desc.stage_count == 1 && code.stage == x2w_gpu::GpuStage::compute &&
            (indirect_light_settings().level > 0 || vbind::compute_capture_on()) &&
            code.bytes && code.size) {
            using Role = x2w_gpu::ambient::Role;
            const std::string_view container(static_cast<const char*>(code.bytes), code.size);
            static const char* const resolve = std::getenv("X2W_LIGHT_RESOLVE_MARKER");
            static const char* const radiance = std::getenv("X2W_LIGHT_RADIANCE_MARKER");
            static const char* const air = std::getenv("X2W_LIGHT_AIR_MARKER");
            if (resolve && *resolve && container.find(resolve) != container.npos) entry->light_role = Role::resolve;
            else if (radiance && *radiance && container.find(radiance) != container.npos) entry->light_role = Role::radiance;
            else if (air && *air && container.find(air) != container.npos) entry->light_role = Role::air;
            if (entry->light_role != Role::none)
                log("x2w-gpu indirect light shader %s role=%u", code.key, unsigned(entry->light_role));
        }
        if (!record && gpu_drop_trace()) {
            if (!entry->missing_keys) entry->missing_keys = std::make_unique<std::string>();
            *entry->missing_keys += std::string(" ") + x2w_gpu::gpu_stage_name(code.stage) + "=" + code.key;
        }
    }
    if (desc.slot == 86) {
        // Keep the parsed graphics state alongside the stage keys recorded at creation.
        if (!desc.desc || desc.desc_bytes < 8 || desc.desc_bytes > 0x10000 || !x2w_local::readable(desc.desc, desc.desc_bytes))
            return false;
        std::vector<std::uint64_t> copy((desc.desc_bytes + 7) / 8);
        std::memcpy(copy.data(), desc.desc, desc.desc_bytes);
        x2w_gpu::VpFormatMap map;
        map.target = &gpu_target_format;
        map.depth = &gpu_plain_format;
        map.input = &gpu_plain_format;
        x2w_gpu::VpStream stream = x2w_gpu::vp_parse_stream(copy.data(), desc.desc_bytes, map, &gpu_readable);
        if (stream.stopped || stream.truncated) {
            static LONG stops;
            if (first_calls(stops, 8))
                log("x2w-gpu stream: subobject %u at +0x%zx %s; the rest is ignored", stream.stopped_kind, stream.stopped_at,
                    stream.truncated ? "runs past the end" : "of unknown size");
        }
        // A compute stream keeps no graphics state (HostPipeline::new takes the compute stage alone).
        const bool compute = desc.stage_count == 1 && desc.stages[0].stage == x2w_gpu::GpuStage::compute;
        if (!compute) entry->base.pending->graphics = std::make_unique<x2w_gpu::VpGraphicsState>(std::move(stream.graphics));
    }
    entry->listed_bad = gpu_warm_bad().contains(desc, entry->base.pending->graphics != nullptr);
    if (entry->exposure_reset && !entry->listed_bad && !entry->base.pending->missing && the_device && !gpu_synchronous_build())
        gpu_queue_request(entry->base, true);
    pipeline->own().gpu[0] = entry.release();
    return true;
}

// The view WP3/WP4 read, made once per host pipeline.
const x2w_gpu::GpuPipeline* gpu_view_for(const std::shared_ptr<const x2w_gpu::VpHostPipeline>& host_ptr,
                                                    const x2w_gpu::VpPending& pending,
                                                    const std::vector<x2w_gpu::GpuStage>& stages)
{
    const x2w_gpu::VpHostPipeline* host = host_ptr.get();
    std::lock_guard<std::mutex> held(gpu_made_lock);
    auto& slot = gpu_made()[host];
    if (slot) return &slot->view;
    auto* made = new GpuMade;
    made->host = host_ptr;
    x2w_gpu::GpuPipeline& v = made->view;
    v.compute = host->compute;
    v.mesh = host->mesh;
    v.rects = host->rects;
    v.pso = host->pso;
    if (pending.graphics) {
        const auto& g = *pending.graphics;
        v.color_mask = x2w_gpu::vp_host_blend(g.blend, g.targets).RenderTarget[0].RenderTargetWriteMask;
        v.depth_test = g.depth.DepthEnable != FALSE;
        v.depth_write = v.depth_test && g.depth.DepthWriteMask == D3D12_DEPTH_WRITE_MASK_ALL;
    }
    v.root = host->root;
    v.stage_count = static_cast<std::uint32_t>((std::min<std::size_t>)(host->stages.size(), std::size(v.stages)));
    const auto slots = [](const std::vector<x2w_gpu::VpRootView>& in, std::vector<x2w_gpu::GpuRootSlot>& out) {
        for (const auto& r : in) out.push_back({r.slot, static_cast<std::uint8_t>(r.param), r.table});
    };
    for (std::uint32_t i = 0; i < v.stage_count; ++i) {
        const x2w_gpu::VpBoundStage& b = host->stages[i];
        x2w_gpu::GpuPipelineStage& s = v.stages[i];
        s.shader = static_cast<const x2w_gpu::GpuShader*>(b.tag);
        s.stage = gpu_stage_of(b.stage);
        s.user_data_param = b.has_user_data ? b.user_data_param : UINT32_MAX;
        s.user_data_dwords = b.user_data_dwords;
        slots(b.root_cbvs, s.root_cbvs);
        slots(b.root_srvs, s.root_srvs);
        slots(b.root_uavs, s.root_uavs);
        for (const auto& r : b.root_srvs)
            if (r.slot == x2w_gpu::abi::slot_strides && v.strides_param == UINT32_MAX) v.strides_param = r.param;
        // DispatchX: the user-data dword the host fills with the thread count along X.
        if (host->compute && s.shader && b.has_user_data)
            for (const auto& u : s.shader->user_data)
                if (u.root == x2w_gpu::GpuRootType::other && u.other == x2w_gpu::abi::user_dispatch_threads_x) {
                    v.threads_x_param = b.user_data_param;
                    v.threads_x_dword = u.index;
                }
    }
    v.vertex_slots = host->slots;
    const x2w_gpu::VpGraphicsState* g = pending.graphics.get();
    for (std::size_t i = 0; i < host->inputs.size(); ++i) {
        const auto& e = host->inputs[i];
        const UINT step = g && i < g->inputs.size() ? g->inputs[i].InstanceDataStepRate : 0;
        v.inputs.push_back({e.slot, e.offset, static_cast<DXGI_FORMAT>(e.format), e.per_instance, step});
    }
    v.topology_type = host->compute ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED
        : host->mesh ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE : x2w_gpu::vp_host_topology_type(host->topology);
    v.title_topology = static_cast<std::uint32_t>(host->topology);
    if (v.title_topology == x2w_gpu::vp_topology_rectangles)
        log("x2w-gpu rectangle PSO %p: %s", static_cast<void*>(v.pso), v.rects ? "geometry stage" : "fallback");
    v.impl = made;
    (void)stages;
    slot = made;
    return &made->view;
}

bool gpu_same_targets(const x2w_gpu::GpuTargets& t, const x2w_gpu::VpGraphicsState& g)
{
    if ((t.read_depth_plane && g.depth.DepthEnable && g.depth.DepthWriteMask != D3D12_DEPTH_WRITE_MASK_ZERO) ||
        (t.read_stencil_plane && g.depth.StencilEnable && g.depth.StencilWriteMask)) return false;
    if (t.count != g.targets.NumRenderTargets || t.dsv != g.depth_format || (t.samples ? t.samples : 1) != g.samples.Count) return false;
    for (std::uint32_t i = 0; i < t.count && i < 8; ++i)
        if (t.rtv[i] != g.targets.RTFormats[i]) return false;
    return true;
}

bool gpu_same(const x2w_gpu::GpuTargets& a, const x2w_gpu::GpuTargets& b)
{
    return a.read_depth_plane == b.read_depth_plane && a.read_stencil_plane == b.read_stencil_plane &&
        a.count == b.count && a.dsv == b.dsv && a.samples == b.samples && std::equal(a.rtv, a.rtv + (std::min)(a.count, 8u), b.rtv);
}

struct GpuCompileWatch {
    const x2w_gpu::VpShaderView* stages;
    std::size_t count;
    DWORD thread;
    ULONGLONG started;
    bool reported = false;

    std::string stage_keys() const
    {
        std::string keys;
        for (std::size_t i = 0; i < count; ++i) {
            const auto* record = static_cast<const x2w_gpu::GpuShader*>(stages[i].tag);
            keys += " stage " + std::to_string(stages[i].stage) + "=" + (record ? record->key : "missing");
        }
        return keys;
    }

    static void CALLBACK elapsed(PTP_CALLBACK_INSTANCE, void* context, PTP_TIMER)
    {
        auto& watch = *static_cast<GpuCompileWatch*>(context);
        static LONG reports;
        if (!first_calls(reports, 48)) return;
        watch.reported = true;
        log("x2w-gpu PSO still building: tid %lu start tick %llu elapsed %llu ms%s",
            watch.thread, watch.started, GetTickCount64() - watch.started, watch.stage_keys().c_str());
    }
};

bool gpu_observe_build(ID3D12Device* device, const x2w_gpu::VpDeviceCaps& caps, const x2w_gpu::VpShaderView* shaders,
                           std::size_t count, const x2w_gpu::VpGraphicsState* graphics, const x2w_gpu::VpMakeOptions& options,
                           x2w_gpu::VpHostPipeline& host, std::string& why)
{
    GpuCompileWatch watch{shaders, count, GetCurrentThreadId(), GetTickCount64()};
    static const bool every_build = x2w_gpu::gpu_env_is("X2W_GPU_PSO_WATCH", '2');
    if (every_build) {
        std::string targets;
        if (graphics) {
            for (UINT i = 0; i < (std::min)(graphics->targets.NumRenderTargets, 8u); ++i)
                targets += " rt" + std::to_string(i) + "=" + std::to_string(graphics->targets.RTFormats[i]);
            targets += " depth=" + std::to_string(graphics->depth_format) + " topology=" + std::to_string(graphics->topology)
                    + " samples=" + std::to_string(graphics->samples.Count);
        }
        log("x2w-gpu PSO begin: tid %lu start tick %llu%s%s", watch.thread, watch.started, watch.stage_keys().c_str(), targets.c_str());
        const bool ok = x2w_gpu::vp_make_host_pipeline(device, caps, shaders, count, graphics, options, host, why);
        log("x2w-gpu PSO end: tid %lu start tick %llu result %s", watch.thread, watch.started, ok ? "success" : why.c_str());
        return ok;
    }
    struct Timer {
        PTP_TIMER handle;
        void stop() {
            if (!handle) return;
            SetThreadpoolTimer(handle, nullptr, 0, 0);
            WaitForThreadpoolTimerCallbacks(handle, TRUE);
            CloseThreadpoolTimer(handle);
            handle = nullptr;
        }
        ~Timer() { stop(); }
    } timer{CreateThreadpoolTimer(GpuCompileWatch::elapsed, &watch, nullptr)};
    if (timer.handle) {
        LARGE_INTEGER delay{};
        delay.QuadPart = -5000ll * 10000;
        FILETIME due{delay.LowPart, static_cast<DWORD>(delay.HighPart)};
        SetThreadpoolTimer(timer.handle, &due, 0, 0);
    }
    const bool ok = x2w_gpu::vp_make_host_pipeline(device, caps, shaders, count, graphics, options, host, why);
    timer.stop();
    if (watch.reported)
        log("x2w-gpu PSO build returned: tid %lu start tick %llu elapsed %llu ms result %s",
            watch.thread, watch.started, GetTickCount64() - watch.started, ok ? "success" : why.c_str());
    return ok;
}

unsigned gpu_pipeline_setting(const char* name, unsigned fallback, unsigned maximum)
{
    char text[24]{};
    const DWORD size = GetEnvironmentVariableA(name, text, sizeof(text));
    if (!size || size >= sizeof(text) || text[0] == '-') return fallback;
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    return end && *end == 0 ? static_cast<unsigned>((std::min<unsigned long>)(maximum, value)) : fallback;
}

GpuJobs& gpu_build_queue()
{
    static GpuJobs* jobs = [] {
        // Workers are process-scoped; a title wrapper may be released while its build is pending.
        HMODULE module = nullptr;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                           reinterpret_cast<LPCWSTR>(&gpu_build_queue), &module);
        gpu_install_compiler_guard();
        const unsigned normal = std::clamp(std::thread::hardware_concurrency() * 2 / 3, 2u, 16u);
        const unsigned workers = (std::max)(1u, gpu_pipeline_setting("X2W_GPU_PIPELINE_WORKERS", normal, 64));
        const unsigned wait = gpu_pipeline_setting("X2W_GPU_PIPELINE_WAIT_MS", 500, 10000);
        const unsigned screen = gpu_pipeline_setting("X2W_GPU_SCREEN_WAIT_MS", 10000, 20000);
        log("x2w-gpu pipeline workers %u + %u screen, shared frame waits %u / %u ms", workers, screen ? 2u : 0u, wait, screen);
        return new GpuJobs(workers, screen ? 2 : 0, GpuJobs::Ms(wait), GpuJobs::Ms(screen), [](bool screen_only) {
            if (!screen_only) SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
        });
    }();
    return *jobs;
}

bool gpu_compile_host(ID3D12Device* device, const x2w_gpu::VpDeviceCaps& caps,
                      const x2w_gpu::VpShaderView* shaders, std::size_t count, const x2w_gpu::VpGraphicsState* graphics,
                      const x2w_gpu::VpMakeOptions& options, x2w_gpu::VpHostPipeline& host, std::string& why)
{
    static const bool watch = x2w_gpu::gpu_env_is("X2W_GPU_PSO_WATCH", '1') ||
                              x2w_gpu::gpu_env_is("X2W_GPU_PSO_WATCH", '2');
    struct Scope {
        Scope() { gpu_in_compiler = true; }
        ~Scope() { gpu_in_compiler = false; }
    } scope;
    const auto make = watch ? gpu_observe_build : x2w_gpu::vp_make_host_pipeline;
    return make(device, caps, shaders, count, graphics, options, host, why);
}

bool gpu_synchronous_build()
{
    static const bool synchronous = x2w_gpu::gpu_env_is("X2W_GPU_PIPELINE_SYNC", '1');
    return synchronous;
}

void gpu_queue_request(GpuRequest& request, bool initialization)
{
    std::call_once(request.requested, [&] {
        const auto options = gpu_make_options();
        Microsoft::WRL::ComPtr<ID3D12Device> device = device_of(the_device);
        const auto caps = gpu_caps();
        auto& pending = *request.pending;
        const bool screen = initialization || (pending.graphics && pending.graphics->screen_pass());
        auto key = x2w_gpu::vp_pipeline_key(pending, options);
        request.job = gpu_build_queue().request(std::move(key), request.pending, screen,
            [device, caps, options](x2w_gpu::VpPending& p) {
                x2w_gpu::vp_ready(p, device.Get(), caps, gpu_pipeline_cache(), options, gpu_compile_host);
            }, initialization);
    });
}

const x2w_gpu::GpuPipeline* gpu_pipeline_ready(Wrapper* pipeline, const GpuTargets* targets, GpuPsoDrop* drop)
{
    auto* entry = pipeline ? static_cast<GpuPipelineEntry*>(pipeline->own().gpu[0]) : nullptr;
    if (!entry || !the_device) return nullptr;
    if (entry->listed_bad) {
        if (drop) drop->status = GpuPsoStatus::warm_bad;
        if (!entry->base.reported.exchange(true)) {
            static LONG reports;
            if (first_calls(reports, 32))
                log("x2w-gpu refused compute pipeline listed in configured warm-bad cache: %s",
                    entry->records[0] ? entry->records[0]->key : "missing shader record");
        }
        return nullptr;
    }
    auto& cache = gpu_pipeline_cache();
    // A depth target given by its typeless texture format (a DSV without a recorded view format) is the D format.
    x2w_gpu::GpuTargets bound{};
    if (targets) {
        bound = *targets;
        if (depth_target_format(bound.dsv) != DXGI_FORMAT_UNKNOWN) bound.dsv = depth_target_format(bound.dsv);
        targets = &bound;
    }
    GpuRequest* request = &entry->base;
    auto* pending = request->pending.get();
    const x2w_gpu::VpGraphicsState* g = pending->graphics.get();
    if (targets && g && !gpu_same_targets(*targets, *g)) {
        std::lock_guard<std::mutex> held(entry->lock);
        request = nullptr;
        for (auto& v : entry->variants)
            if (gpu_same(v.first, *targets)) request = v.second.get();
        if (!request) {
            // A copy of the title pipeline with the bound formats.
            auto variant = std::make_unique<GpuRequest>();
            variant->pending->creation_slot = entry->base.pending->creation_slot;
            for (const auto& s : entry->base.pending->shaders) variant->pending->add(s);
            auto state = std::make_unique<x2w_gpu::VpGraphicsState>(*g);
            state->targets = {};
            for (std::uint32_t i = 0; i < targets->count && i < 8; ++i) state->targets.RTFormats[i] = targets->rtv[i];
            state->targets.NumRenderTargets = (std::min)(targets->count, 8u);
            state->depth_format = targets->dsv;
            if (targets->read_depth_plane) state->depth.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
            if (targets->read_stencil_plane) state->depth.StencilWriteMask = 0;
            state->samples.Count = targets->samples ? targets->samples : 1;
            if (state->samples.Count == 1) state->samples.Quality = 0;
            variant->pending->graphics = std::move(state);
            request = variant.get();
            entry->variants.emplace_back(*targets, std::move(variant));
            static LONG variants;
            if (first_calls(variants, 8))
                log("x2w-gpu pipeline variant for bound targets (%u targets, first %d, depth %d; the stream named %u, first %d, depth %d)",
                    targets->count, targets->count ? static_cast<int>(targets->rtv[0]) : 0, static_cast<int>(targets->dsv),
                    g->targets.NumRenderTargets, static_cast<int>(g->targets.RTFormats[0]), static_cast<int>(g->depth_format));
        }
    }
    pending = request->pending.get();
    if (gpu_synchronous_build() || pending->missing) {
        x2w_gpu::vp_ready(*pending, device_of(the_device), gpu_caps(), cache, gpu_make_options(), gpu_compile_host);
    } else {
        auto& jobs = gpu_build_queue();
        const bool screen = pending->graphics && pending->graphics->screen_pass();
        gpu_queue_request(*request, entry->exposure_reset);
        const auto frame = static_cast<unsigned>(InterlockedCompareExchange(&frame_number, 0, 0));
        const bool ready = entry->exposure_reset ? jobs.ready_initialization(request->job) : jobs.ready(request->job, frame, screen);
        if (!ready) {
            if (entry->exposure_reset && !request->timeout_reported.exchange(true))
                log("x2w-gpu exposure reset PSO timed out at first use, frame=%u variant=%d",
                    frame, int(request != &entry->base));
            if (drop) drop->status = GpuPsoStatus::compiling;
            return nullptr;
        }
        if (request->job->failure) {
            if (drop) drop->status = GpuPsoStatus::worker_failure;
            if (!request->reported.exchange(true)) {
                try { std::rethrow_exception(request->job->failure); }
                catch (const std::exception& e) { log("x2w-gpu pipeline worker failed: %s", e.what()); }
                catch (...) { log("x2w-gpu pipeline worker failed with an unknown C++ exception"); }
            }
            return nullptr;
        }
        pending = request->job->state.get();
    }
    const auto* host = pending->made.get();
    const bool fresh = !request->reported.exchange(true, std::memory_order_relaxed);
    if (!host) {
        if (drop) {
            drop->status = pending->missing ? GpuPsoStatus::missing_shader : GpuPsoStatus::refused;
            drop->detail = pending->why.c_str();
        }
        if (fresh) {
            // Once per reason, with the keys of the first few.
            static std::mutex seen_lock;
            static std::vector<std::string> seen;
            static LONG refusals;
            bool first = false;
            {
                std::lock_guard<std::mutex> held(seen_lock);
                if (std::find(seen.begin(), seen.end(), pending->why) == seen.end()) {
                    seen.push_back(pending->why);
                    first = true;
                }
            }
            if (first || first_calls(refusals, 16)) {
                std::string keys;
                for (std::size_t i = 0; i < entry->records.size(); ++i) {
                    keys += ' ';
                    keys += x2w_gpu::gpu_stage_name(entry->stages[i]);
                    keys += ' ';
                    keys += entry->records[i] ? entry->records[i]->key : "missing";
                }
                log("x2w-gpu no host pipeline (slot %u):%s: %s", pending->creation_slot, keys.c_str(), pending->why.c_str());
            }
        }
        return nullptr;
    }
    if (fresh) {
        static LONG made;
        if (first_calls(made, 16))
            log("x2w-gpu host %s pipeline made (slot %u; %llu made, %llu shared, %llu refused so far)",
                host->compute ? "compute" : host->mesh ? "mesh" : "graphics", pending->creation_slot,
                static_cast<unsigned long long>(cache.made_count.load()), static_cast<unsigned long long>(cache.shared_count.load()),
                static_cast<unsigned long long>(cache.refused_count.load()));
    }
    return gpu_view_for(pending->made, *pending, entry->stages);
}

void gpu_pipeline_released(Wrapper* pipeline)
{
    if (!pipeline) return;
    auto* entry = static_cast<GpuPipelineEntry*>(pipeline->own().gpu[0]);
    pipeline->own().gpu[0] = nullptr;
    // The host pipelines stay in the cache (a recorded draw may still use them); the record goes.
    delete entry;
}
