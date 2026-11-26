// Mirror shader-visible view and sampler heaps with title descriptor memory. Slot i mirrors title slot i. Refresh
// on view creation (slots 17–19, 22, 61, 70 and 71), copies (23 and 24), explicit lookup and submission. Submission
// catches records copied by the title as raw bytes. Samplers and placed buffers use X2WV or X2WP records so their
// metadata survives such copies.
//
// Queue newly visible textures for upload and revisit one sixteenth at submission. gpu_heap_upload_due performs
// uploads on the draw path where a recording list is available.

namespace vbind { std::optional<bool> prepare_packed_light(Wrapper*, Wrapper*, ID3D12Resource*, x2w_packed_light::BoundCopy* = nullptr); }
namespace gpu_detail_heap {

using gpu_mirror::HostHeap;

struct Mirror {
    Wrapper* title_heap = nullptr;
    u64 title_cpu = 0, title_gpu = 0;
    std::uint32_t count = 0, step = 0;
    bool sampler = false;
    std::unique_ptr<HostHeap> host;
};

SRWLOCK lock = SRWLOCK_INIT;
std::vector<Mirror*>& mirrors()
{
    static std::vector<Mirror*> m;
    return m;
}
// Released heaps: kept (a submission may still read through their descriptors); few are ever released.
std::vector<Mirror*>& gone()
{
    static std::vector<Mirror*> g;
    return g;
}

inline GpuHostHeap* opaque(HostHeap* h) { return reinterpret_cast<GpuHostHeap*>(h); }
inline HostHeap* host_of(GpuHostHeap* h) { return reinterpret_cast<HostHeap*>(h); }

// The mirror whose title memory holds title CPU handle `handle`, and the slot.
Mirror* mirror_at(u64 handle, std::uint32_t* index)
{
    Mirror* found = nullptr;
    AcquireSRWLockShared(&lock);
    for (Mirror* m : mirrors())
        if (handle >= m->title_cpu && handle < m->title_cpu + u64{m->count} * m->step) {
            found = m;
            if (index) *index = static_cast<std::uint32_t>((handle - m->title_cpu) / m->step);
            break;
        }
    ReleaseSRWLockShared(&lock);
    return found;
}
Mirror* mirror_of(const HostHeap* h)
{
    Mirror* found = nullptr;
    AcquireSRWLockShared(&lock);
    for (Mirror* m : mirrors())
        if (m->host.get() == h) found = m;
    ReleaseSRWLockShared(&lock);
    return found;
}

// ---- Env from X2W ----------------------------------------------------------------------------------------------

bool env_find(void*, u64 address, u64 bytes, ID3D12Resource** resource, u64* offset, u64* left)
{
    GpuFound f;
    if (gpu_find(address, bytes, &f) && f.resource) {
        *resource = f.resource;
        *offset = f.offset;
        *left = f.bytes_left;
        return true;
    }
    // Until WP3's gpu_find answers: the window that holds the whole range, bounded by the range.
    ID3D12Resource* window = nullptr;
    u64 at = 0;
    if (!desktop_range(address, bytes, &window, &at) || !window) return false;
    *resource = window;
    *offset = at;
    *left = bytes;
    return true;
}

bool env_record(void*, std::uint32_t id, gpu_mirror::Record* out)
{
    ViewRecord r{};
    bool have = false;
    AcquireSRWLockShared(&note_lock);
    if (id < view_records().size()) {
        r = view_records()[id];
        out->materialized = view_materialized()[id] != 0;
        have = true;
    }
    ReleaseSRWLockShared(&note_lock);
    if (!have || r.slot < 17 || r.slot > 22 || r.slot == 20 || r.slot == 21) return false;
    out->id = id;
    out->slot = r.slot;
    out->resource = r.resource;
    out->counter = r.counter;
    out->desc_bytes = r.desc_bytes;
    std::memcpy(out->desc, r.desc, sizeof(out->desc));
    // The description as view_slot gives it to the desktop: console formats mapped, depth views fixed.
    if (r.slot == 18 || r.slot == 19) {
        // (A resource the title released has no desktop resource to fix the view for.)
        Wrapper* const w = r.resource ? as_wrapper(reinterpret_cast<const void*>(r.resource)) : nullptr;
        ID3D12Resource* const desktop = w && InterlockedCompareExchange(&w->own().refs, 0, 0) > 0
            ? reinterpret_cast<ID3D12Resource*>(unwrap(r.resource)) : nullptr;
        if (r.desc_bytes >= 4) {
            DXGI_FORMAT format;
            std::memcpy(&format, out->desc, sizeof(format));
            format = desktop_format(format);
            std::memcpy(out->desc, &format, sizeof(format));
        }
        if (r.slot == 18 && desktop && reinterpret_cast<u64>(desktop) != r.resource) {
            D3D12_SHADER_RESOURCE_VIEW_DESC d{};
            if (r.desc_bytes >= sizeof(d)) {
                std::memcpy(&d, out->desc, sizeof(d));
                fix_srv(d, desktop);
                std::memcpy(out->desc, &d, sizeof(d));
            } else if (default_depth_srv(desktop, d)) {
                std::memcpy(out->desc, &d, sizeof(d));
                out->desc_bytes = sizeof(d);
            }
        }
    }
    return true;
}

bool env_resource(void*, u64 wrapper, gpu_mirror::ResourceFacts* out)
{
    Wrapper* w = as_wrapper(reinterpret_cast<const void*>(wrapper));
    if (!w || InterlockedCompareExchange(&w->own().refs, 0, 0) <= 0) return false;
    auto* const desktop = static_cast<ID3D12Resource*>(w->own().real);
    if (!desktop) return false;
    const D3D12_RESOURCE_DESC d = desktop->GetDesc();
    out->desktop = desktop;
    out->buffer = d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER;
    out->title_memory = reinterpret_cast<u64>(w->own().title_memory);
    const ResourceInfo* info = resource_info(w);
    constexpr auto both = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    out->depth_and_uav = info && (info->console.Flags & both) == both;
    out->bytes = info && info->console.Width ? info->console.Width : d.Width;
    return true;
}

using DepthPair = std::shared_ptr<x2w_depth_twins::Twins>;
std::mutex depth_lock;
std::unordered_map<Wrapper*, DepthPair>& depth_pairs()
{
    static std::unordered_map<Wrapper*, DepthPair> pairs;
    return pairs;
}
DepthPair depth_uav(u64 wrapper)
{
    auto* w = as_wrapper(reinterpret_cast<const void*>(wrapper));
    std::lock_guard<std::mutex> guard(depth_lock);
    if (!w || InterlockedCompareExchange(&w->own().refs, 0, 0) <= 0) return {};
    auto& pairs = depth_pairs();
    if (auto it = pairs.find(w); it != pairs.end()) return it->second;
    const auto* info = resource_info(w);
    auto pair = std::make_shared<x2w_depth_twins::Twins>();
    if (!info || !pair->create(device_of(the_device), static_cast<ID3D12Resource*>(w->own().real), info->console)) pair.reset();
    pairs.emplace(w, pair);
    return pair;
}


bool env_upload(void*, void* list, u64 wrapper, ID3D12Resource* texture)
{
    Wrapper* w = as_wrapper(reinterpret_cast<const void*>(wrapper));
    auto* l = static_cast<Wrapper*>(list);
    if (!w || !l || !recording(l)) return false;
    if (auto result = vbind::prepare_packed_light(l, w, texture)) return *result;
    bool entered_copy_dest = false;
    const bool ok = upload_texture(l, w, texture, &entered_copy_dest);
    // A heap-indexed read has no per-table binding that can restore the resource after the copy.
    // Early exits before COPY_DEST (including GPU-written/UAV textures) preserve the existing state.
    if (entered_copy_dest) require_state(l, texture, x2w_gpu::texture_read_state(recording(l)->GetType()));
    return ok;
}

u64 late_texture(const std::uint32_t* words, D3D12_SHADER_RESOURCE_VIEW_DESC& srv, bool& pending, bool allow_create)
{
    D3D12_RESOURCE_DESC desc{};
    if (!x2w_late_texture::decode(words, desc, srv)) { pending = false; return 0; }
    const u64 address = u64{words[1]} | (u64{words[2]} << 32);
    if (!address) return 0;
    using x2w_late_texture::Resolution;
    Resolution why{};
    Wrapper* texture = x2w_late_texture::resolve([&] {
        if (auto* native = native_tiled_bc_texture(address, desc, srv)) return native;
        Wrapper* w = existing_memory_texture(address, desc);
        if (w && w->own().real) return w;
        return packed_light_source(address, desc) ? publish_linear_texture(the_device, address, desc, true, false, 0, allow_create) : nullptr;
    }, [&] {
        // Small mips begin at the start of the allocation. If its first byte is unavailable, skip building the detile
        // layout.
        if (!x2w_local::descriptor_memory_readable(address, 1)) return 8u;
        return detile_decline(desc, srv, address);
    },
    [&] {
        const bool streamed = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && desc.MipLevels > 1;
        return publish_linear_texture(the_device, address, desc, true, streamed, streamed ? deferred_host_base(desc, srv) : 0, allow_create);
    }, why);
    if (allow_create) {
        static LONG outcomes[5], attempts;
        InterlockedIncrement(&outcomes[static_cast<unsigned>(why)]);
        const LONG attempt = InterlockedIncrement(&attempts);
        if ((attempt & (attempt - 1)) == 0)
            log("x2w-gpu deferred texture retries %ld: existing host %ld, bytes unavailable %ld, layout refused %ld, "
                "publish failed %ld, published %ld", attempt, outcomes[0], outcomes[1], outcomes[2], outcomes[3], outcomes[4]);
    }
    pending = why == Resolution::unavailable;
    if (pending && !texture) texture = partial_mip_texture(desc, srv, address, pending, allow_create);
    if (!allow_create) {
        static LONG lookups, hits;
        if (texture) InterlockedIncrement(&hits);
        const LONG n = InterlockedIncrement(&lookups);
        if ((n & (n - 1)) == 0) log("x2w-gpu material texture budget lookups %ld, reused %ld", n, hits);
    }
    if (!texture) return 0;
    if (texture->own().partial_mip_srv) {
        const auto native = static_cast<ID3D12Resource*>(texture->own().real)->GetDesc();
        x2w_resident_mips::rebase(srv, desc.MipLevels - native.MipLevels);
    }
    srv.Format = desktop_format(srv.Format);
    fix_srv(srv, static_cast<ID3D12Resource*>(texture->own().real));
    static LONG recovered;
    const LONG n = InterlockedIncrement(&recovered);
    if ((n & (n - 1)) == 0)
        log("x2w-gpu deferred texture lookups recovered %ld: 0x%llx %llux%u format %d", n, address,
            desc.Width, desc.Height, desc.Format);
    return reinterpret_cast<u64>(texture);
}

// The runtime owns a single desktop device. Sample the local budget without retaining its device.
u64 lazy_local_room(ID3D12Device* device)
{
    using Microsoft::WRL::ComPtr;
    static const ComPtr<IDXGIAdapter3> adapter = [device] {
        ComPtr<IDXGIFactory4> factory;
        ComPtr<IDXGIAdapter3> result;
        if (device && SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
            factory->EnumAdapterByLuid(device->GetAdapterLuid(), IID_PPV_ARGS(&result));
        return result;
    }();
    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    if (adapter && SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)) && info.Budget) {
        const auto ceiling = info.Budget - info.Budget / 10;
        return ceiling > info.CurrentUsage ? ceiling - info.CurrentUsage : 0;
    }
    std::lock_guard lock(linear_srv_mutex);
    constexpr u64 ceiling = 4ull << 30;
    return ceiling > linear_srv_bytes ? ceiling - linear_srv_bytes : 0;
}

gpu_mirror::Env env_for(Wrapper* device)
{
    gpu_mirror::Env e;
    e.device = device ? device_of(device) : nullptr;
    e.mapping_epoch = gpu_mapping_generation;
    e.copied_bytes = []() -> u64 { return x2w_startup::SliceFootprints::recorded_bytes; };
    e.allocated_bytes = []() -> u64 { return thread_linear_allocations; };
    e.local_room = lazy_local_room;
    e.allocation_size = [](ID3D12Device* device, const std::uint32_t* words) -> u64 {
        D3D12_RESOURCE_DESC guest{};
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        if (!x2w_late_texture::decode(words, guest, srv)) return UINT64_MAX;
        const auto native = x2w_resident_mips::host_desc(sanitized(guest), deferred_host_base(guest, srv));
        return device->GetResourceAllocationInfo(0, 1, &native).SizeInBytes;
    };
    e.coherent_window = [](u64 address, u64 bytes, ID3D12Resource* expected, u64 offset) {
        u64 begin = 0, end = 0;
        if (!known_private(address, &begin, &end) || bytes > end - address) return false;
        ID3D12Resource* current = nullptr;
        u64 at = 0;
        return desktop_range(address, bytes, &current, &at) && current == expected && at == offset;
    };
    e.find = env_find;
    e.record = env_record;
    e.resource = env_resource;
    e.depth_uav = depth_uav;
    static const bool retry = x2w_local::env_on("X2W_GPU_LATE_TEXTURES");
    if (retry) {
        e.late_texture = late_texture;
        e.placement = [](u64 wrapper, const D3D12_SHADER_RESOURCE_VIEW_DESC& srv, std::uint32_t (&words)[8]) {
            auto* w = reinterpret_cast<Wrapper*>(wrapper);
            if (!w->own().title_memory) return false;
            auto* info = resource_info(w);
            if (!info) return false;
            auto guest = srv;
            if (w->own().partial_mip_srv && guest.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2D) {
                const auto native = static_cast<ID3D12Resource*>(w->own().real)->GetDesc();
                const UINT base = info->console.MipLevels - native.MipLevels;
                guest.Texture2D.MostDetailedMip += base;
                if (guest.Texture2D.ResourceMinLODClamp > 0)
                    guest.Texture2D.ResourceMinLODClamp += float(base);
            }
            return x2w_late_texture::encode(info->console, guest, reinterpret_cast<u64>(w->own().title_memory), words);
        };
    }
    e.upload = env_upload;
    return e;
}

void sink(const char* text) { log("%s", text); }

// probe_view_limit, once: 2^22 views with twins (X2W_GPU_TWINS=0: none), else 2^21, else 1,000,000.
const gpu_mirror::Limits& limits(ID3D12Device* device)
{
    static const gpu_mirror::Limits l = [device] {
        const auto found = gpu_mirror::probe_view_limit(device, x2w_gpu::gpu_twins_on());
        log("x2w-gpu heap: host view heaps hold %u views, twins %s (from %u); view indices are clamped to the null "
            "slot %u", found.raw ? 2 * found.raw : found.views, found.raw ? "on" : "off", found.raw, found.null());
        if (x2w_gpu::gpu_twins_on() && !found.raw)
            log("x2w-gpu heap: the host takes no heap of 2^22 views: no raw twins, but the export expects them "
                "(raw_views=2^21); shaders that pick a twin read the null slot");
        return found;
    }();
    return l;
}

void info_of(const Mirror* m, GpuHeapInfo* out)
{
    if (!out) return;
    *out = {};
    out->heap = opaque(m->host.get());
    out->title_heap = m->title_heap;
    out->title_gpu = m->title_gpu;
    out->title_cpu = m->title_cpu;
    out->count = m->count;
    out->sampler = m->sampler;
    out->host = m->host->heap();
}

void view_info(const gpu_mirror::View& v, GpuViewInfo* out)
{
    if (!out) return;
    *out = {};
    using x2w_gpu::GpuViewKind;
    switch (v.kind) {
    case gpu_mirror::Kind::constant: out->kind = GpuViewKind::cbv; break;
    case gpu_mirror::Kind::shader: out->kind = GpuViewKind::srv; break;
    case gpu_mirror::Kind::unordered: out->kind = GpuViewKind::uav; break;
    case gpu_mirror::Kind::sampler: out->kind = GpuViewKind::sampler; break;
    case gpu_mirror::Kind::placed: out->kind = v.unordered ? GpuViewKind::uav : GpuViewKind::srv; break;
    default: out->kind = GpuViewKind::none; break;
    }
    out->texture = v.is_texture();
    if (out->texture && v.has_desc && out->kind == GpuViewKind::srv) {
        out->view_format = v.srv.Format;
        switch (v.srv.ViewDimension) {
        case D3D12_SRV_DIMENSION_TEXTURE1D: out->first_mip = v.srv.Texture1D.MostDetailedMip; break;
        case D3D12_SRV_DIMENSION_TEXTURE1DARRAY:
            out->first_mip = v.srv.Texture1DArray.MostDetailedMip;
            out->first_slice = v.srv.Texture1DArray.FirstArraySlice; break;
        case D3D12_SRV_DIMENSION_TEXTURE2D:
            out->first_mip = v.srv.Texture2D.MostDetailedMip;
            out->plane = v.srv.Texture2D.PlaneSlice; break;
        case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:
            out->first_mip = v.srv.Texture2DArray.MostDetailedMip;
            out->plane = v.srv.Texture2DArray.PlaneSlice;
            out->first_slice = v.srv.Texture2DArray.FirstArraySlice; break;
        case D3D12_SRV_DIMENSION_TEXTURE3D: out->first_mip = v.srv.Texture3D.MostDetailedMip; break;
        case D3D12_SRV_DIMENSION_TEXTURECUBE: out->first_mip = v.srv.TextureCube.MostDetailedMip; break;
        case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY:
            out->first_mip = v.srv.TextureCubeArray.MostDetailedMip;
            out->first_slice = v.srv.TextureCubeArray.First2DArrayFace; break;
        default: break;
        }
    }
    out->buffer = v.is_buffer();
    out->title_resource = reinterpret_cast<const void*>(v.wrapper);
    out->desktop = v.resource;
    out->depth_twins = v.depth_twins;
    const auto e = gpu_mirror::view_elements(v);
    if (v.kind == gpu_mirror::Kind::constant) {
        out->title_address = v.address;
        out->bytes = v.bytes;
    } else if (e && v.has_desc) {
        const bool uav = v.kind == gpu_mirror::Kind::unordered || (v.kind == gpu_mirror::Kind::placed && v.unordered);
        const u64 unit = uav ? gpu_mirror::element_size(v.uav) : gpu_mirror::element_size(v.srv);
        const u64 first = uav ? v.uav.Buffer.FirstElement : v.srv.Buffer.FirstElement;
        const std::uint32_t n = uav ? v.uav.Buffer.NumElements : v.srv.Buffer.NumElements;
        out->title_address = v.address + first * unit;
        out->bytes = u64{n} * unit;
        out->stride = gpu_mirror::stride_word(e);
    }
}

// The record at title slot `handle`, as mirror reads it (8 dwords; a sampler heap's 4).
bool read_record(const Mirror* m, u64 handle, std::uint32_t words[8])
{
    std::memset(words, 0, 32);
    const std::size_t bytes = (std::min<std::size_t>)(m->step, 32);
    return gpu_mirror::read_memory(gpu_mirror::Env{}, handle, words, bytes);
}

// Device slot 61's description (27 words) as an X2WV sampler record at `dest`, which the title's copies then carry.
void note_console_sampler(Wrapper* device, u64 desc, u64 dest)
{
    if (!desc || !x2w_local::readable(reinterpret_cast<const void*>(desc), 27 * 4)) return;
    // The destination can be a CPU-only staging heap. Such heaps have no shader-visible
    // mirror, but CopyDescriptors must see a valid desktop sampler and its 16-byte title token.
    AcquireSRWLockShared(&note_lock);
    const bool in_sampler_heap = std::any_of(heap_infos().begin(), heap_infos().end(), [&](const HeapInfo& h) {
        constexpr u64 step = x2w_gpu::abi::sampler_increment;
        return h.type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER && h.title_cpu && dest >= h.title_cpu &&
               (dest - h.title_cpu) % step == 0 && (dest - h.title_cpu) / step < h.count;
    });
    ReleaseSRWLockShared(&note_lock);
    if (!in_sampler_heap) return;
    std::uint32_t words[27];
    std::memcpy(words, reinterpret_cast<const void*>(desc), sizeof(words));
    const D3D12_SAMPLER_DESC d = gpu_mirror::console_sampler(words);
    view_slot<22, 1>(device, reinterpret_cast<u64>(&d), dest, 0, 0);
}

// Device slots 70 and 71: a buffer view of memory that X2W's handler left with no record the mirror can follow (it
// refused it, or wrote nothing) gets an X2WP record of the view.
template <class Input>
void note_placed(Mirror* m, const D3D12_RESOURCE_DESC* resource, const Input* input, u64 dest, bool unordered)
{
    if (!resource || !input || !x2w_local::readable(resource, sizeof(*resource)) || !x2w_local::readable(input, sizeof(*input)))
        return;
    if (resource->Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || static_cast<int>(input->view.ViewDimension) != 1 ||
        !input->address)
        return;
    std::uint32_t words[8];
    if (!read_record(m, dest, words)) return;
    gpu_mirror::View v;
    if (gpu_mirror::view_of_record(env_for(the_device), words, false, v)) return;  // X2W's record stands
    if (input->view.Buffer.FirstElement > 0xffffffffu || !input->view.Buffer.NumElements) return;
    gpu_mirror::PlacedRecord p;
    p.address = input->address;
    p.first = static_cast<std::uint32_t>(input->view.Buffer.FirstElement);
    p.elements = input->view.Buffer.NumElements;
    p.stride = input->view.Buffer.StructureByteStride;
    p.format = static_cast<std::uint32_t>(desktop_format(input->view.Format));
    p.raw = (input->view.Buffer.Flags & 1) != 0;
    p.unordered = unordered;
    p.width = static_cast<std::uint32_t>((std::min<u64>)(resource->Width, 0xffffffffu));
    gpu_mirror::placed_words(p, words);
    if (!x2w_local::writable(reinterpret_cast<void*>(dest), 32)) return;
    std::memcpy(reinterpret_cast<void*>(dest), words, (std::min<std::size_t>)(m->step, 32));
}

}  // namespace gpu_detail_heap

// ---- gpu_hooks.h: WP5 -------------------------------------------------------------------------------------

void gpu_heap_created(Wrapper* device, Wrapper* heap, const D3D12_DESCRIPTOR_HEAP_DESC& desc, std::uint64_t title_cpu,
                          std::uint64_t title_gpu)
{
    using namespace gpu_detail_heap;
    const bool views = desc.Type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    const bool samplers = desc.Type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    if (!(views || samplers) || !(desc.Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) || !title_cpu || !heap || !device) return;
    static const bool logged = (HostHeap::env_log() = sink, gpu_mirror::fit_log() = sink, true);
    (void)logged;
    const std::uint32_t step = views ? x2w_gpu::abi::view_increment : x2w_gpu::abi::sampler_increment;
    const gpu_mirror::Env e = env_for(device);
    auto host = HostHeap::create(e, samplers, desc.NumDescriptors, limits(e.device), title_cpu, step);
    if (!host) {
        log("x2w-gpu heap: no mirror heap for the title's %s heap of %u", views ? "view" : "sampler", desc.NumDescriptors);
        return;
    }
    const gpu_mirror::Layout& l = host->layout();
    log("x2w-gpu heap: mirror of the title's %s heap at 0x%llx: %u title slots%s, %u dynamic, null slot %u, twins from %u, "
        "%u host views", views ? "view" : "sampler", title_cpu, l.count, l.capped ? " (capped)" : "", l.dynamic, l.null, l.raw,
        l.size);
    auto* m = new Mirror;
    m->title_heap = heap;
    m->title_cpu = title_cpu;
    m->title_gpu = title_gpu;
    m->count = l.count;
    m->step = step;
    m->sampler = samplers;
    m->host = std::move(host);
    AcquireSRWLockExclusive(&lock);
    mirrors().push_back(m);
    ReleaseSRWLockExclusive(&lock);
    heap->own().gpu[0] = m;
}

void gpu_heap_released(Wrapper* heap)
{
    using namespace gpu_detail_heap;
    auto* m = static_cast<Mirror*>(heap->own().gpu[0]);
    if (!m) return;
    heap->own().gpu[0] = nullptr;
    AcquireSRWLockExclusive(&lock);
    auto& all = mirrors();
    all.erase(std::remove(all.begin(), all.end(), m), all.end());
    gone().push_back(m);
    ReleaseSRWLockExclusive(&lock);
}

void gpu_heap_view(Wrapper* device, int slot, std::uint64_t a, std::uint64_t b, std::uint64_t c, std::uint64_t d)
{
    using namespace gpu_detail_heap;
    u64 dest = 0;
    switch (slot) {
    case 17: case 22: case 61: dest = b; break;
    case 18: case 70: case 71: dest = c; break;
    case 19: dest = d; break;
    default: return;  // 20, 21: render targets and depth are not in shader-visible heaps
    }
    if (slot == 61) note_console_sampler(device, a, dest);
    std::uint32_t index = 0;
    Mirror* m = mirror_at(dest, &index);
    if (!m) return;
    if (slot == 70)
        note_placed(m, reinterpret_cast<const D3D12_RESOURCE_DESC*>(a), reinterpret_cast<const MemoryShaderResourceView*>(b), dest, false);
    if (slot == 71)
        note_placed(m, reinterpret_cast<const D3D12_RESOURCE_DESC*>(a), reinterpret_cast<const MemoryUnorderedAccessView*>(b), dest, true);
    m->host->mirror(index, false);
}

void gpu_heap_copy_simple(Wrapper*, UINT count, std::uint64_t dest, std::uint64_t, D3D12_DESCRIPTOR_HEAP_TYPE)
{
    using namespace gpu_detail_heap;
    std::uint32_t index = 0;
    if (Mirror* m = mirror_at(dest, &index)) m->host->mirror_range(index, count);
}

void gpu_heap_copy(Wrapper*, UINT dest_count, const D3D12_CPU_DESCRIPTOR_HANDLE* dest_starts, const UINT* dest_sizes, UINT,
                       const D3D12_CPU_DESCRIPTOR_HANDLE*, const UINT*, D3D12_DESCRIPTOR_HEAP_TYPE)
{
    using namespace gpu_detail_heap;
    if (!dest_starts) return;
    for (UINT i = 0; i < dest_count; ++i) {
        std::uint32_t index = 0;
        if (Mirror* m = mirror_at(dest_starts[i].ptr, &index)) m->host->mirror_range(index, dest_sizes ? dest_sizes[i] : 1);
    }
}

void gpu_heap_submit(ID3D12CommandQueue*, UINT, Wrapper* const*)
{
    using namespace gpu_detail_heap;
    std::vector<Mirror*> all;
    AcquireSRWLockShared(&lock);
    all = mirrors();
    ReleaseSRWLockShared(&lock);
    for (Mirror* m : all) m->host->submit();
}

bool gpu_heap_for(Wrapper* title_heap, GpuHeapInfo* out)
{
    using namespace gpu_detail_heap;
    if (!title_heap) return false;
    auto* m = static_cast<Mirror*>(title_heap->own().gpu[0]);
    if (!m) return false;
    info_of(m, out);
    return true;
}

bool gpu_heap_at(std::uint64_t title_gpu_handle, GpuHeapInfo* out)
{
    using namespace gpu_detail_heap;
    Mirror* found = nullptr;
    AcquireSRWLockShared(&lock);
    for (Mirror* m : mirrors())
        if (m->title_gpu && title_gpu_handle >= m->title_gpu && title_gpu_handle < m->title_gpu + u64{m->count} * m->step) found = m;
    ReleaseSRWLockShared(&lock);
    if (!found) return false;
    info_of(found, out);
    return true;
}

bool gpu_heap_view_at(GpuHostHeap* heap, std::uint32_t index, GpuViewInfo* out)
{
    using namespace gpu_detail_heap;
    auto* h = host_of(heap);
    if (!h || index >= h->count()) return false;
    Mirror* m = mirror_of(h);
    if (!m) return false;
    std::uint32_t words[8];
    if (!read_record(m, m->title_cpu + u64{index} * m->step, words)) return false;
    gpu_mirror::View v;
    if (!h->view_at(index, words, v)) return false;
    view_info(v, out);
    return true;
}

u64 gpu_heap_title_address_at(GpuHostHeap* heap, std::uint32_t index)
{
    using namespace gpu_detail_heap;
    auto* host = host_of(heap);
    auto* m = host && index < host->count() ? mirror_of(host) : nullptr;
    std::uint32_t words[8];
    if (!m || m->sampler || !read_record(m, m->title_cpu + u64{index} * m->step, words)) return 0;
    const auto env = env_for(the_device);
    u64 address = 0;
    if (gpu_mirror::title_as_address(env, words, &address)) {
        static LONG samples;
        const auto n = InterlockedIncrement(&samples);
        if (n <= 4 || (n & (n-1)) == 0)
            log("x2w-gpu AS title address: frame=%ld count=%ld index=%u address=%llx", frame_number, n, index, address);
        return address;
    }
    gpu_mirror::View v;
    if (!gpu_mirror::view_of_record(env, words, false, v)) return 0;
    GpuViewInfo view{};
    view_info(v, &view);
    return view.buffer ? view.title_address : 0;
}

ID3D12Resource* gpu_heap_kept_at(GpuHostHeap* heap, std::uint32_t index)
{
    auto* host = gpu_detail_heap::host_of(heap);
    return host ? host->kept_at(index) : nullptr;
}

void gpu_material_slots(GpuHostHeap* heap, std::vector<x2w_material_trace::Slot>& slots)
{
    if (auto* host = gpu_detail_heap::host_of(heap)) host->material_slots(slots);
}

x2w_material_trace::Record gpu_material_record(GpuHostHeap* heap, UINT slot, UINT base, x2w_material::Mask fields)
{
    auto* h = gpu_detail_heap::host_of(heap);
    return h ? h->material_record(slot, base, fields) : x2w_material_trace::Record{};
}
unsigned gpu_materialize(Wrapper* list, GpuHostHeap* heap, const x2w_material_trace::Words& words, x2w_material::Mask fields)
{
    auto* h = gpu_detail_heap::host_of(heap);
    return h ? h->materialize(list, words, fields) : 0;
}

bool gpu_heap_mirror(GpuHostHeap* heap, std::uint32_t index, std::uint64_t title_handle, bool, GpuViewInfo* out)
{
    using namespace gpu_detail_heap;
    auto* h = host_of(heap);
    if (!h || index >= h->count()) return false;
    if (title_handle) {
        // (The handle the caller read the table at names the slot.)
        std::uint32_t at = 0;
        Mirror* m = mirror_at(title_handle, &at);
        if (m && m->host.get() == h) index = at;
    }
    const bool shown = h->mirror(index);
    if (out && gpu_heap_view_at(heap, index, out) && out->texture) out->desktop = h->kept_at(index);
    return shown;
}

bool gpu_heap_dynamic(GpuHostHeap* heap, std::uint32_t n, std::uint32_t* first)
{
    auto* h = gpu_detail_heap::host_of(heap);
    if (!h) return false;
    const auto got = h->dynamic(n);
    if (!got) return false;
    if (first) *first = *got;
    return true;
}

void gpu_heap_release(GpuHostHeap* heap, std::uint32_t first, std::uint32_t n)
{
    if (auto* h = gpu_detail_heap::host_of(heap)) h->release(first, n);
}

bool gpu_heap_make_listed(GpuHostHeap* heap, std::uint32_t source, const std::uint32_t (&record)[8], std::uint32_t index,
                              GpuViewInfo* out)
{
    using namespace gpu_detail_heap;
    auto* h = host_of(heap);
    if (!h) return false;
    gpu_mirror::View v;
    if (!h->make_listed(index, record, source, &v)) return false;
    view_info(v, out);
    if (out && out->texture) out->desktop = h->kept_at(index);
    return true;
}

void gpu_heap_make_null(GpuHostHeap* heap, std::uint32_t index)
{
    if (auto* h = gpu_detail_heap::host_of(heap)) h->make_null(index);
}

void gpu_heap_raw_at(GpuHostHeap* heap, std::uint32_t index, ID3D12Resource* buffer, std::uint64_t offset, std::uint64_t bytes)
{
    if (auto* h = gpu_detail_heap::host_of(heap)) h->raw_at(index, buffer, offset, bytes);
}

void gpu_heap_raw_uav_at(GpuHostHeap* heap, std::uint32_t index, ID3D12Resource* buffer, std::uint64_t offset,
                             std::uint64_t bytes)
{
    if (auto* h = gpu_detail_heap::host_of(heap)) h->raw_uav_at(index, buffer, offset, bytes);
}

void gpu_heap_null_raw_at(GpuHostHeap* heap, std::uint32_t index)
{
    if (auto* h = gpu_detail_heap::host_of(heap)) h->null_raw_at(index);
}

void gpu_heap_null_raw_uav_at(GpuHostHeap* heap, std::uint32_t index)
{
    if (auto* h = gpu_detail_heap::host_of(heap)) h->null_raw_uav_at(index);
}

void gpu_heap_constant_at(GpuHostHeap* heap, std::uint32_t index, ID3D12Resource* buffer, std::uint64_t gpu, std::uint32_t bytes)
{
    if (auto* h = gpu_detail_heap::host_of(heap)) h->constant_at(index, buffer, gpu, bytes);
}

D3D12_GPU_DESCRIPTOR_HANDLE gpu_heap_gpu_handle(GpuHostHeap* heap, std::uint32_t index)
{
    auto* h = gpu_detail_heap::host_of(heap);
    return h ? h->gpu_handle(index) : D3D12_GPU_DESCRIPTOR_HANDLE{};
}

std::uint32_t gpu_heap_count(GpuHostHeap* heap)
{
    auto* h = gpu_detail_heap::host_of(heap);
    return h ? h->count() : 0;
}

ID3D12DescriptorHeap* gpu_heap_host(GpuHostHeap* heap)
{
    auto* h = gpu_detail_heap::host_of(heap);
    return h ? h->heap() : nullptr;
}

std::uint64_t gpu_heap_strides_gpu(GpuHostHeap* heap)
{
    auto* h = gpu_detail_heap::host_of(heap);
    return h ? h->strides_gpu() : 0;
}

// Declared in gpu_hooks.h (vhooks fe1f8bd31, bool result): PLAN.md 4b, the textures shaders pick by index that
// are due, up to `max`, uploaded on `list` (a recording title list, the draw's: upload_texture needs one). True when
// at least one was uploaded.
bool gpu_heap_upload_due(Wrapper* list, std::size_t max)
{
    using namespace gpu_detail_heap;
    auto* command = list ? recording(list) : nullptr;
    if (!command) return false;
    const auto type = command->GetType();
    if (type != D3D12_COMMAND_LIST_TYPE_DIRECT && type != D3D12_COMMAND_LIST_TYPE_COMPUTE) return false;
    std::vector<Mirror*> all;
    AcquireSRWLockShared(&lock);
    all = mirrors();
    ReleaseSRWLockShared(&lock);
    std::size_t done = 0;
    for (Mirror* m : all)
        if (!m->sampler && done < max) done += m->host->upload_due(list, max - done);
    return done != 0;
}
