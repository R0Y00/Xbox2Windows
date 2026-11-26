// Included after the resource creation and descriptor handlers.
struct MemoryShaderResourceView {
    D3D12_SHADER_RESOURCE_VIEW_DESC view;
    u64 address;
};
static_assert(offsetof(MemoryShaderResourceView, address) == 0x28);

bool same_texture_description(const D3D12_RESOURCE_DESC& a, const D3D12_RESOURCE_DESC& b)
{
    return a.Dimension == b.Dimension && a.Alignment == b.Alignment && a.Width == b.Width &&
           a.Height == b.Height && a.DepthOrArraySize == b.DepthOrArraySize && a.MipLevels == b.MipLevels &&
           a.Format == b.Format && a.SampleDesc.Count == b.SampleDesc.Count &&
           a.SampleDesc.Quality == b.SampleDesc.Quality && a.Layout == b.Layout && a.Flags == b.Flags;
}

bool tiled_any_on()
{
    static const bool on = [] {
        const bool v = x2w_local::env_on("X2W_MEMORY_SRV_TILED_ANY");
        if (v) log("x2w-memsrv X2W_MEMORY_SRV_TILED_ANY=1: single-mip tiled memory SRVs of any 16-byte BC format and size");
        return v;
    }();
    return on;
}
bool bc16_format(DXGI_FORMAT f)
{
    switch (f) {
    case DXGI_FORMAT_BC2_UNORM: case DXGI_FORMAT_BC2_UNORM_SRGB: case DXGI_FORMAT_BC3_UNORM: case DXGI_FORMAT_BC3_UNORM_SRGB:
    case DXGI_FORMAT_BC5_UNORM: case DXGI_FORMAT_BC5_SNORM: case DXGI_FORMAT_BC6H_UF16: case DXGI_FORMAT_BC6H_SF16:
    case DXGI_FORMAT_BC7_UNORM: case DXGI_FORMAT_BC7_UNORM_SRGB:
        return true;
    default:
        return false;
    }
}
bool tiled_bc_srv(const D3D12_RESOURCE_DESC& r)
{
    if (tiled_any_on() && bc16_format(r.Format) && static_cast<UINT>(r.Layout) == 0x105 && r.Alignment == 0 &&
        (r.Flags == D3D12_RESOURCE_FLAG_NONE || static_cast<UINT>(r.Flags) == 0x300000) && r.Width && r.Height &&
        r.Width <= D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION && r.Height <= D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION)
        return true;
    return r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && r.DepthOrArraySize == 1 &&
           r.MipLevels == 1 && r.SampleDesc.Count == 1 && r.SampleDesc.Quality == 0 &&
           (r.Format == DXGI_FORMAT_BC3_UNORM || r.Format == DXGI_FORMAT_BC7_UNORM) &&
           static_cast<UINT>(r.Layout) == 0x105 && r.Alignment == 0 &&
           (r.Flags == D3D12_RESOURCE_FLAG_NONE || static_cast<UINT>(r.Flags) == 0x300000) &&
           r.Width <= D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION && r.Height <= D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION &&
           // A single mip pads its pitch and height to whole 4 KiB tiles (Bc16Surface); the desktop
           // texture needs whole 4x4 blocks.
           r.Width && r.Height && r.Width % 4 == 0 && r.Height % 4 == 0;
}

// One line for the first publication of a view whose last tile row or column is partial. A published
// view gets a content summary, so a title run can tell console-tiled data from zeros; a held one names
// why it stays refused.
void log_partial_tile_view(const D3D12_RESOURCE_DESC& r, u64 address, const char* held, bool published)
{
    static LONG lines, unpublished;
    const auto bytes = x2w_startup::Bc16Surface(r).size();
    if (!published) {
        if (first_calls(unpublished, 8))
            log("x2w-d3d12 Device::slot70 partial-tile BC view %llux%u format %d layout 0x%x: %s, %llu tiled bytes",
                r.Width, r.Height, r.Format, static_cast<unsigned>(r.Layout), held ? held : "texture_unavailable", bytes);
        return;
    }
    if (!first_calls(lines, 64)) return;
    const auto* data = reinterpret_cast<const unsigned char*>(address);
    const auto nonzero = [&](u64 from) {
        unsigned count = 0;
        for (u64 i = from; i < from + 4096; ++i) count += data[i] != 0;
        return count;
    };
    // publish_tiled_texture already found the whole range readable.
    log("x2w-d3d12 Device::slot70 partial-tile BC view %llux%u format %d layout 0x%x: published, %llu tiled bytes, "
        "nonzero first/last 4 KiB %u/%u hash %016llx", r.Width, r.Height, r.Format, static_cast<unsigned>(r.Layout),
        bytes, nonzero(0), nonzero(bytes - 4096), hash_memory(data, static_cast<std::size_t>(bytes)));
}

// The size the XG stand-in reports for a description: desktop copyable footprints rounded to 64 KiB.
u64 xg_reported_bytes(Wrapper* device, const D3D12_RESOURCE_DESC& desc)
{
    const x2w_startup::SliceFootprints plan(device_of(device), sanitized(desc), 1);
    return plan.size == UINT64_MAX ? 0 : (plan.size + 0xFFFF) & ~u64{0xFFFF};
}

bool supports_placed_memory_srv(const D3D12_RESOURCE_DESC& r, const D3D12_SHADER_RESOURCE_VIEW_DESC& v);
bool upstream_console_srv_on()
{
    static const bool on = x2w_local::env_on("X2W_UPSTREAM_CONSOLE_SRV");
    return on;
}
// The shapes slot 70 took before address_texture.h, or a texture in a console layout gfx10_layout.h
// addresses with a view that stays inside it.
bool supports_memory_srv(const D3D12_RESOURCE_DESC& r, const D3D12_SHADER_RESOURCE_VIEW_DESC& v)
{
    if (supports_placed_memory_srv(r, v)) return true;
    return console_layout_srv(r) && x2w_startup::address_view_fits(r, v);
}

bool supports_placed_memory_srv(const D3D12_RESOURCE_DESC& r, const D3D12_SHADER_RESOURCE_VIEW_DESC& v)
{
    if (r.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
        return x2w_startup::supports_buffer_memory_srv(r, v);
    if (x2w_startup::bc64::supports(r)) {
        if (v.Format != r.Format || v.Shader4ComponentMapping != D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING)
            return false;
        if (v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2DARRAY) {
            const auto& a = v.Texture2DArray;
            return r.DepthOrArraySize == 2 && a.FirstArraySlice < r.DepthOrArraySize && a.ArraySize > 0 &&
                   a.ArraySize <= r.DepthOrArraySize - a.FirstArraySlice && a.MostDetailedMip < r.MipLevels &&
                   a.MipLevels > 0 && a.MipLevels <= r.MipLevels - a.MostDetailedMip &&
                   a.PlaneSlice == 0 && a.ResourceMinLODClamp == 0;
        }
        return r.DepthOrArraySize == 1 && v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2D &&
               v.Texture2D.MostDetailedMip < r.MipLevels && v.Texture2D.MipLevels > 0 &&
               v.Texture2D.MipLevels <= r.MipLevels - v.Texture2D.MostDetailedMip &&
               !v.Texture2D.PlaneSlice && v.Texture2D.ResourceMinLODClamp == 0;
    }
    if (x2w_startup::supports_single_element_srv(r, v)) return true;
    if (x2w_startup::supports_volume_memory_srv(r, v)) return true;
    return r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && r.Width && r.Height &&
           r.DepthOrArraySize == 1 && r.MipLevels == 1 && r.SampleDesc.Count == 1 && r.SampleDesc.Quality == 0 &&
           (r.Format == DXGI_FORMAT_R8_UNORM || tiled_bc_srv(r)) && v.Format == r.Format && !(r.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) &&
           v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2D &&
           v.Shader4ComponentMapping == D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING &&
           v.Texture2D.MostDetailedMip == 0 && v.Texture2D.MipLevels == 1 && v.Texture2D.PlaneSlice == 0 &&
           v.Texture2D.ResourceMinLODClamp == 0.0f;
}

// The layout tables describe the bytes independently of the desktop adapter's format support.
bool desktop_address_view_supported(Wrapper* device, const D3D12_RESOURCE_DESC& r,
                                    const D3D12_SHADER_RESOURCE_VIEW_DESC& v)
{
    auto* const desktop = device_of(device);
    D3D12_FEATURE_DATA_FORMAT_INFO planes{r.Format};
    if (FAILED(desktop->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, &planes, sizeof(planes))) || planes.PlaneCount != 1)
        return false;

    D3D12_FEATURE_DATA_FORMAT_SUPPORT view{v.Format};
    if (FAILED(desktop->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &view, sizeof(view))) ||
        !(view.Support1 & (D3D12_FORMAT_SUPPORT1_SHADER_LOAD | D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE)))
        return false;

    D3D12_FEATURE_DATA_FORMAT_SUPPORT storage{r.Format};
    if (r.Format == v.Format) storage = view;
    else if (FAILED(desktop->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &storage, sizeof(storage)))) return false;
    const UINT req = r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? D3D12_FORMAT_SUPPORT1_TEXTURE3D
                                                                       : D3D12_FORMAT_SUPPORT1_TEXTURE2D;
    return (storage.Support1 & req) &&
           (r.MipLevels == 1 || (storage.Support1 & D3D12_FORMAT_SUPPORT1_MIP));
}

bool memory_srv_destination(u64 handle)
{
    bool valid = false;
    AcquireSRWLockShared(&note_lock);
    for (const auto& heap : heap_infos()) {
        if (heap.type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV || !heap.increment) continue;
        const u64 base = heap.title_cpu ? heap.title_cpu : heap.desktop_cpu;
        if (handle >= base && (handle - base) % heap.increment == 0 &&
            (handle - base) / heap.increment < heap.count) {
            valid = true;
            break;
        }
    }
    ReleaseSRWLockShared(&note_lock);
    return valid;
}

// With X2W_RESOURCE_INDEX_CHECK=1 (or 2, see resource_index_check_mode), every memory SRV call (a full pass, for tests) checks that each live resource with title
// memory is in the address index (entries are never removed, so it stays there after a release). A miss is logged and ends the process with code 3; this is a test mode only.
void check_resource_address_index()
{
    if (resource_index_check_mode() == 0) return;
    AcquireSRWLockShared(&note_lock);
    std::optional<u64> missing;
    for (const auto& [wrapper, info] : resource_infos()) {
        const u64 base = reinterpret_cast<u64>(wrapper->own().title_memory);
        if (!base || InterlockedCompareExchange(&wrapper->own().refs, 0, 0) <= 0 || resource_address_index().contains(base, wrapper))
            continue;
        missing = base;
        break;
    }
    const std::size_t entries = resource_address_index().size();
    ReleaseSRWLockShared(&note_lock);
    if (!missing) return;
    log("x2w-d3d12 resource address index misses a live resource at 0x%llx (index %zu entries)", *missing, entries);
    ExitProcess(3);
}

Wrapper* existing_memory_texture(u64 address, const D3D12_RESOURCE_DESC& desc)
{
    X2W_ZONE("srv existing_memory_texture scan");
    Wrapper* match = nullptr;
    AcquireSRWLockShared(&note_lock);
    // A null address never gets here (memory_srv_refusal), and the index holds no null base.
    resource_address_index().for_each_reaching(address, 1, [&](const Wrapper* wrapper) {
        const auto& info = resource_infos().at(wrapper);
        if (reinterpret_cast<u64>(wrapper->own().title_memory) != address ||
            wrapper->own().tiled_address_srv ||
            InterlockedCompareExchange(&wrapper->own().refs, 0, 0) <= 0 ||
            !same_texture_description(desc, info.console)) return false;
        if (match) {
            match = nullptr;
            return true;
        }
        match = const_cast<Wrapper*>(wrapper);
        return false;
    });
    ReleaseSRWLockShared(&note_lock);
    return match;
}

// An unmatched or interior view must not reinterpret bytes owned by a live resource.
bool address_resource_overlap(u64 address, u64 span)
{
    X2W_ZONE("srv address_resource_overlap scan");
    bool found = false;
    AcquireSRWLockShared(&note_lock);
    resource_address_index().for_each_reaching(address, span, [&](const Wrapper* wrapper) {
        if (wrapper->own().tiled_address_srv || InterlockedCompareExchange(&wrapper->own().refs, 0, 0) <= 0)
            return false;
        const auto& info = resource_infos().at(wrapper);
        const u64 bytes = info.bytes ? info.bytes
            : info.console.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER ? info.console.Width : 0;
        return found = x2w_startup::address_span_touches(address, span, reinterpret_cast<u64>(wrapper->own().title_memory), bytes);
    });
    ReleaseSRWLockShared(&note_lock);
    return found;
}

// A live render-target, depth or UAV resource overlapping the texel is written on the desktop side, so its
// title memory does not hold the texel. For a resource of unknown size, its base is the only known byte.
bool gpu_written_overlap(u64 address, UINT texel_bytes)
{
    X2W_ZONE("srv gpu_written_overlap scan");
    constexpr UINT gpu_written = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL |
                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    bool found = false;
    AcquireSRWLockShared(&note_lock);
    resource_address_index().for_each_reaching(address, texel_bytes, [&](const Wrapper* wrapper) {
        const auto& info = resource_infos().at(wrapper);
        const u64 bytes = info.bytes ? info.bytes : info.console.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER ? info.console.Width : 0;
        return found = x2w_startup::address_span_touches(address, texel_bytes, reinterpret_cast<u64>(wrapper->own().title_memory), bytes) &&
            (static_cast<UINT>(info.console.Flags) & gpu_written) && InterlockedCompareExchange(&wrapper->own().refs, 0, 0) > 0;
    });
    ReleaseSRWLockShared(&note_lock);
    return found;
}

Wrapper* existing_memory_buffer(Wrapper* device, u64 address, u64 bytes)
{
    X2W_ZONE("srv existing_memory_buffer scan");
    if (!address || !bytes) return nullptr;
    const auto placed = placed_buffer_range(device, address, bytes);
    if (placed.reject) return nullptr;
    if (placed.owner) {
        if (InterlockedCompareExchange(&placed.owner->own().refs, 0, 0) <= 0) return nullptr;
        const auto* info = resource_info(placed.owner);
        if (!info || info->console.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
            (info->console.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE))
            return nullptr;
        return const_cast<Wrapper*>(placed.owner);
    }
    static const int index_mode = [] {
        char v[16]{};
        const DWORD n = GetEnvironmentVariableA("X2W_BUFFER_ADDRESS_INDEX", v, sizeof(v));
        const int mode = n && n < sizeof(v) ? (std::strcmp(v, "1") == 0 ? 1 : std::strcmp(v, "check") == 0 ? 2 : 0) : 0;
        if (mode) log("x2w-memsrv X2W_BUFFER_ADDRESS_INDEX=%s: memory buffer lookups use the address index%s", v,
                      mode == 2 ? ", checked against the walk on every call" : "");
        return mode;
    }();
    // The walk's filters, applied to each candidate either way.
    const auto eligible = [](const Wrapper* wrapper, const ResourceInfo& info) {
        return !(!wrapper->own().real && info.console.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) &&
               info.placement == Placement::none && info.console.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER &&
               !(info.console.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) &&
               InterlockedCompareExchange(&const_cast<Wrapper*>(wrapper)->own().refs, 0, 0) > 0;
    };
    Wrapper* indexed = nullptr;
    if (index_mode) {
        AcquireSRWLockShared(&note_lock);
        bool ambiguous = false;
        buffer_address_index().containing(address, bytes, [&](const Wrapper* wrapper, u64 base, u64) {
            const auto found = resource_infos().find(wrapper);
            if (found == resource_infos().end() || !eligible(wrapper, found->second)) return true;
            // As the walk: the record's current base and width decide (the index holds them as noted).
            const u64 now = reinterpret_cast<u64>(wrapper->own().title_memory), width = found->second.console.Width;
            if (now != base || address < now || address - now >= width || bytes > width - (address - now)) return true;
            if (indexed) {
                ambiguous = true;
                return false;
            }
            indexed = const_cast<Wrapper*>(wrapper);
            return true;
        });
        ReleaseSRWLockShared(&note_lock);
        if (ambiguous) indexed = nullptr;
        if (index_mode == 1) return indexed;
    }
    Wrapper* match = nullptr;
    AcquireSRWLockShared(&note_lock);
    resource_address_index().for_each_reaching(address, 1, [&](const Wrapper* wrapper) {
        const auto& info = resource_infos().at(wrapper);
        if (!wrapper->own().real && info.console.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) return false;
        if (info.placement != Placement::none ||
            info.console.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
            (info.console.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) ||
            InterlockedCompareExchange(&wrapper->own().refs, 0, 0) <= 0)
            return false;
        const u64 base = reinterpret_cast<u64>(wrapper->own().title_memory);
        const u64 width = info.console.Width;
        if (!base || address < base || address - base >= width || bytes > width - (address - base))
            return false;
        if (match) {
            match = nullptr;
            return true;
        }
        match = const_cast<Wrapper*>(wrapper);
        return false;
    });
    ReleaseSRWLockShared(&note_lock);
    if (index_mode == 2) {
        static LONG64 checked, differed;
        const LONG64 n = InterlockedIncrement64(&checked);
        const bool same = indexed == match;
        const LONG64 d = same ? differed : InterlockedIncrement64(&differed);
        static LONG shown;
        if (!same && first_calls(shown, 8))
            log("x2w-memsrv buffer address index DIFFERS at 0x%llx (%llu bytes): index %p, walk %p", address, bytes, static_cast<void*>(indexed),
                static_cast<void*>(match));
        if ((n & (n - 1)) == 0) log("x2w-memsrv buffer address index checked %lld lookups against the walk: %lld differ", n, d);
    }
    return match;
}

// Backing for unplaced buffer memory SRVs: 64 KiB page-aligned placed resource on an existing heap
// (OpenExistingHeapFromAddress zero-copy aliasing on hardware, allocations up to 256 MiB) or committed UPLOAD
// buffer (fallback).
struct UnplacedBufferEntry {
    Wrapper* device = nullptr;
    void* allocation_base = nullptr;
    u64 page_start = 0;
    u64 page_span = 0;
    bool is_upload = false;
    x2w_startup::ComPtr<ID3D12Resource> resource;
    Wrapper* wrapper = nullptr;
};

std::mutex unplaced_buffer_mutex;
std::vector<UnplacedBufferEntry> unplaced_buffers;

void clear_unplaced_buffers(Wrapper* device)
{
    std::lock_guard lock(unplaced_buffer_mutex);
    for (auto it = unplaced_buffers.begin(); it != unplaced_buffers.end();) {
        if (it->device == device) {
            if (it->wrapper) {
                retire_placement(it->wrapper);
                if (it->wrapper->own().budget) it->wrapper->own().budget->end_use();
                if (it->wrapper->own().backing_budget) it->wrapper->own().backing_budget->end_use();
                it->wrapper->own().real = nullptr;
                InterlockedDecrement(&it->wrapper->own().refs);
            }
            it->resource.Reset();
            it = unplaced_buffers.erase(it);
        } else {
            ++it;
        }
    }
}

Wrapper* unplaced_memory_buffer(Wrapper* device, u64 address, u64 span_bytes, const D3D12_RESOURCE_DESC& requested_desc)
{
    if (address_resource_overlap(address, span_bytes)) return nullptr;
    u64 page_start = 0, page_span = 0;
    if (!x2w_startup::unplaced_buffer_extent(address, span_bytes, page_start, page_span))
        return nullptr;

    x2w_startup::PlacedBufferRange memory{};
    if (!x2w_startup::measure_placed_buffer(reinterpret_cast<void*>(page_start), page_span, memory) ||
        !memory.covered)
        return nullptr;

    const auto nearby = placed_buffer_range(device, page_start, page_span, true);
    if (nearby.reject || nearby.owner) return nullptr;

    auto* desktop = device_of(device);
    if (!desktop) return nullptr;

    std::lock_guard lock(unplaced_buffer_mutex);
    for (auto& entry : unplaced_buffers) {
        if (entry.device == device && entry.allocation_base == memory.base &&
            x2w_startup::unplaced_buffer_covers(entry.page_start, entry.page_span, address, span_bytes)) {
            x2w_startup::PlacedBufferRange verify{};
            if (!x2w_startup::measure_placed_buffer(reinterpret_cast<void*>(entry.page_start), entry.page_span, verify) ||
                !verify.covered || verify.base != memory.base) {
                continue;
            }
            if (entry.is_upload && entry.resource) {
                void* mapped = nullptr;
                D3D12_RANGE read_range{0, 0};
                if (SUCCEEDED(entry.resource->Map(0, &read_range, &mapped))) {
                    std::memcpy(mapped, reinterpret_cast<const void*>(entry.page_start), static_cast<std::size_t>(entry.page_span));
                    entry.resource->Unmap(0, nullptr);
                }
            }
            return entry.wrapper;
        }
    }

    x2w_startup::ComPtr<ID3D12Resource> native_resource;
    bool is_upload = false;
    ID3D12Heap* heap = nullptr;
    x2w_budget::Ticket* backing_budget = nullptr;

    D3D12_FEATURE_DATA_EXISTING_HEAPS existing_support{};
    const bool existing_heaps_supported =
        SUCCEEDED(desktop->CheckFeatureSupport(D3D12_FEATURE_EXISTING_HEAPS, &existing_support, sizeof(existing_support))) &&
        existing_support.Supported;

    if (existing_heaps_supported && memory.type == MEM_PRIVATE) {
        std::lock_guard placed_lock(placed_mutex);
        for (const auto& [known, h] : existing_heaps) {
            if (known == memory.base) {
                heap = h.real;
                backing_budget = h.budget;
                break;
            }
        }
        if (!heap && x2w_startup::unplaced_existing_heap_allowed(x2w_startup::allocation_bytes(memory.base))) {
            ID3D12Device3* device3 = nullptr;
            if (SUCCEEDED(desktop->QueryInterface(IID_PPV_ARGS(&device3)))) {
                if (SUCCEEDED(device3->OpenExistingHeapFromAddress(memory.base, IID_PPV_ARGS(&heap)))) {
                    backing_budget = x2w_budget::backing_heap(heap, 0);
                    existing_heaps.emplace_back(memory.base, ExistingHeap{heap, backing_budget});
                }
                device3->Release();
            }
        }
        if (heap) {
            const u64 base_u64 = reinterpret_cast<u64>(memory.base);
            const u64 heap_offset = page_start - base_u64;
            const u64 heap_size = heap->GetDesc().SizeInBytes;
            if (heap_offset <= heap_size && page_span <= heap_size - heap_offset) {
                D3D12_RESOURCE_DESC desc{};
                desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
                desc.Width = page_span;
                desc.Height = 1;
                desc.DepthOrArraySize = 1;
                desc.MipLevels = 1;
                desc.Format = DXGI_FORMAT_UNKNOWN;
                desc.SampleDesc.Count = 1;
                desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;

                desktop->CreatePlacedResource(heap, heap_offset, &desc, D3D12_RESOURCE_STATE_COMMON,
                                              nullptr, IID_PPV_ARGS(&native_resource));
            }
        }
    }

    if (!native_resource) {
        D3D12_HEAP_PROPERTIES props{D3D12_HEAP_TYPE_UPLOAD};
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = page_span;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = D3D12_RESOURCE_FLAG_NONE;

        if (SUCCEEDED(desktop->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc,
                                                      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                      IID_PPV_ARGS(&native_resource)))) {
            is_upload = true;
            void* mapped = nullptr;
            D3D12_RANGE read_range{0, 0};
            if (SUCCEEDED(native_resource->Map(0, &read_range, &mapped))) {
                std::memcpy(mapped, reinterpret_cast<const void*>(page_start), static_cast<std::size_t>(page_span));
                native_resource->Unmap(0, nullptr);
            }
        }
    }

    if (!native_resource) return nullptr;

    Wrapper* w = wrap(native_resource.Get(), resource_kind);
    w->own().title_memory = reinterpret_cast<void*>(page_start);
    D3D12_RESOURCE_DESC rdesc{};
    rdesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rdesc.Width = page_span;
    rdesc.Height = 1;
    rdesc.DepthOrArraySize = 1;
    rdesc.MipLevels = 1;
    rdesc.Format = DXGI_FORMAT_UNKNOWN;
    rdesc.SampleDesc.Count = 1;
    rdesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rdesc.Flags = is_upload ? D3D12_RESOURCE_FLAG_NONE : D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;

    note_resource(w, &rdesc, page_span);
    if (!is_upload) {
        w->own().budget = x2w_budget::resource(desktop, native_resource.Get(), x2w_budget::placed);
        w->own().backing_budget = backing_budget;
        if (backing_budget) backing_budget->begin_use();
    } else {
        w->own().budget = x2w_budget::committed(desktop, native_resource.Get(), D3D12_HEAP_TYPE_UPLOAD);
    }

    unplaced_buffers.push_back(UnplacedBufferEntry{device, memory.base, page_start, page_span, is_upload, native_resource, w});
    return w;
}

// Keep metadata for published tiled SRVs; allocate GPU memory only when a
// translated draw actually samples one. A later write to the same address is
// picked up by the uploader even when the description and view token are reused.
std::mutex tiled_srv_mutex;
struct TiledAddressTexture {
    Wrapper* device;
    Wrapper* resource;
    D3D12_RESOURCE_DESC desc;
    x2w_startup::ComPtr<ID3D12Resource> desktop;
};
std::unordered_map<Wrapper*, TiledAddressTexture> tiled_srv_textures;
std::unordered_multimap<u64, Wrapper*> tiled_srv_addresses;

bool partial_tiled_bc(const D3D12_RESOURCE_DESC& desc)
{
    return tiled_bc_srv(desc) && (desc.Width % 64 || desc.Height % 64);
}

bool latest_texture_write_is_rows(u64 address)
{
    AcquireSRWLockShared(&written_lock);
    const auto found = written_textures().find(address);
    const bool rows = found != written_textures().end() && found->second.desktop_rows;
    ReleaseSRWLockShared(&written_lock);
    return rows;
}

// A texture slot 70 reads in a console layout of gfx10_layout.h: not one of the shapes it read before
// (the single texel, the 0x105 BC3/BC7 surface and the 0x109 BC chains keep their own paths).
bool console_layout_srv(const D3D12_RESOURCE_DESC& desc)
{
    return upstream_console_srv_on() && x2w_startup::console_address_texture(desc) && !x2w_startup::single_element_bytes(desc) &&
           !tiled_bc_srv(desc) && !x2w_startup::bc64::supports(desc);
}

// One line for each new kind of slot-70 texture call (format, layout, shape class, view and outcome), so
// a run shows every shape a title asks for without a line per texture.
void log_address_kind(const char* outcome, const D3D12_RESOURCE_DESC& r, const D3D12_SHADER_RESOURCE_VIEW_DESC& v, u64 address)
{
    static std::mutex mutex;
    static std::unordered_map<u64, LONG> kinds;
    const u64 words[] = {static_cast<u64>(r.Format), static_cast<u64>(r.Layout), r.MipLevels > 1 ? 2u : 1u,
                         r.DepthOrArraySize > 1 ? 2u : 1u, static_cast<u64>(r.Flags), static_cast<u64>(r.Dimension),
                         static_cast<u64>(v.Format), static_cast<u64>(v.ViewDimension), reinterpret_cast<u64>(outcome)};
    const u64 key = fnv_bytes(words, sizeof(words));
    std::lock_guard lock(mutex);
    if (kinds.size() >= 200 || kinds[key]++) return;
    log("x2w-d3d12 slot70 texture kind %zu: %s; first seen res{dim=%d fmt=%d size=%llux%ux%u mips=%u layout=0x%x flags=0x%x "
        "align=%llu} view{dim=%d fmt=%d map=0x%x} address=0x%llx", kinds.size(), outcome, r.Dimension, r.Format, r.Width,
        r.Height, r.DepthOrArraySize, r.MipLevels, static_cast<UINT>(r.Layout), static_cast<UINT>(r.Flags), r.Alignment,
        v.ViewDimension, v.Format, v.Shader4ComponentMapping, address);
}

// Bytes the texels of a by-address texture span at their address.
u64 address_texture_bytes(Wrapper* device, const D3D12_RESOURCE_DESC& desc)
{
    if (const UINT element = x2w_startup::single_element_bytes(desc)) return element;
    if (x2w_startup::bc64::supports(desc)) return x2w_startup::bc64::Chain(desc).bytes();
    if (console_layout_srv(desc)) return x2w_startup::gfx10::Layout(desc).bytes();
    if (desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D) {
        if (device) {
            const x2w_startup::SliceFootprints plan(device_of(device), sanitized(desc), desc.MipLevels ? desc.MipLevels : 1);
            if (plan.size != UINT64_MAX && plan.size) return plan.size;
        }
        return x2w_startup::volume_texture_bytes(desc);
    }
    return x2w_startup::Bc16Surface(desc).size();
}

// Why a console-layout description is not read from an address, or null. The two cases a partial-tile
// BC view is held for:
// - the latest XG stand-in copy to the address wrote desktop-footprint rows (only the opt-in BC7 4 KiB
//   copy records a tiled write; the 0x109 BC chains, which XG writes in their layout, keep their own
//   path and never come through here);
// - the layout runs past the size the XG stand-in reports for the description (the desktop copyable
//   footprints of a slice rounded up to 64 KiB, times the slices), which the title may not own.
const char* console_address_hold(Wrapper* device, u64 address, const D3D12_RESOURCE_DESC& desc)
{
    if (latest_texture_write_is_rows(address)) return "console_layout_xg_rows";
    const x2w_startup::SliceFootprints plan(device_of(device), sanitized(desc), desc.MipLevels);
    if (plan.size == UINT64_MAX || !plan.size) return "console_layout_without_footprints";
    const u64 reported = ((plan.size + 0xFFFF) & ~u64{0xFFFF}) * desc.DepthOrArraySize;
    return x2w_startup::gfx10::Layout(desc).bytes() > reported ? "console_layout_beyond_xg_size" : nullptr;
}

Wrapper* publish_tiled_texture(Wrapper* device, u64 address, const D3D12_RESOURCE_DESC& desc)
{
    const auto bytes = address_texture_bytes(device, desc);
    // The transient upload ring accepts individual copies up to 64 MiB.
    if (bytes > (64ull << 20) || !x2w_local::readable(reinterpret_cast<const void*>(address), bytes)) return nullptr;
    std::lock_guard lock(tiled_srv_mutex);
    const auto [first, last] = tiled_srv_addresses.equal_range(address);
    for (auto it = first; it != last; ++it) {
        const auto& known = tiled_srv_textures.at(it->second);
        if (known.device == device && same_texture_description(known.desc, desc)) {
            AcquireSRWLockExclusive(&written_lock);
            ++written_textures()[address].generation;
            ReleaseSRWLockExclusive(&written_lock);
            return known.resource;
        }
    }
    Wrapper* resource = wrap(nullptr, resource_kind);
    resource->own().title_memory = reinterpret_cast<void*>(address);
    resource->own().tiled_address_srv = true;
    note_resource(resource, &desc, bytes);
    tiled_srv_textures.emplace(resource, TiledAddressTexture{device, resource, desc, {}});
    tiled_srv_addresses.emplace(address, resource);
    return resource;
}

ID3D12Resource* sampled_address_texture(Wrapper* resource)
{
    if (!resource || resource->own().kind != resource_kind) return nullptr;
    warm_texture(resource);
    memcut_revive(resource);
    if (!resource->own().tiled_address_srv) return static_cast<ID3D12Resource*>(resource->own().real);
    std::lock_guard lock(tiled_srv_mutex);
    auto& sampled = tiled_srv_textures.at(resource);
    // An already-published token can outlive a write in a different layout.
    if ((partial_tiled_bc(sampled.desc) || console_layout_srv(sampled.desc)) &&
        latest_texture_write_is_rows(reinterpret_cast<u64>(resource->own().title_memory)))
        return nullptr;
    if (!sampled.desktop) {
        const auto desc = sanitized(sampled.desc);
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
        const HRESULT hr = device_of(sampled.device)->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&sampled.desktop));
        if (FAILED(hr)) {
            static LONG failures;
            if (first_calls(failures, 8)) log("x2w-d3d12 failed to materialize tiled memory SRV: 0x%08lx", hr);
            return nullptr;
        }
        x2w_budget::committed(device_of(sampled.device), sampled.desktop.Get(), D3D12_HEAP_TYPE_DEFAULT);
        x2w_startup::mark_packed_stand_in(sampled.desktop.Get(), desktop_format(sampled.desc.Format));
    }
    return sampled.desktop.Get();
}

bool memory_srv_linear_on()
{
    static const bool on = [] {
        const bool v = x2w_local::env_on("X2W_MEMORY_SRV_LINEAR");
        if (v) log("x2w-memsrv X2W_MEMORY_SRV_LINEAR=1: plain 2D memory SRVs get a desktop texture uploaded from the view address");
        return v;
    }();
    return on;
}
// view_slot does not check mip, array or plane ranges, and desktop removes the device on a view it
// rejects, so the range is checked here (MipLevels ~0u means "all from the most detailed mip").
bool linear_view_in_range(const D3D12_RESOURCE_DESC& r, const D3D12_SHADER_RESOURCE_VIEW_DESC& v)
{
    const UINT mips = r.MipLevels, slices = r.DepthOrArraySize;
    const auto mip_range = [&](UINT most, UINT count, float clamp) {
        return most < mips && (count == ~0u || (count > 0 && count <= mips - most)) && clamp >= 0.0f && clamp <= static_cast<float>(mips);
    };
    if (v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2D)
        return mip_range(v.Texture2D.MostDetailedMip, v.Texture2D.MipLevels, v.Texture2D.ResourceMinLODClamp) && v.Texture2D.PlaneSlice == 0;
    if (v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2DARRAY) {
        const auto& a = v.Texture2DArray;
        return mip_range(a.MostDetailedMip, a.MipLevels, a.ResourceMinLODClamp) && a.PlaneSlice == 0 && a.FirstArraySlice < slices &&
               a.ArraySize > 0 && a.ArraySize <= slices - a.FirstArraySlice;
    }
    if (v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURECUBE)
        return r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && r.Width == r.Height && slices == 6 &&
               mip_range(v.TextureCube.MostDetailedMip, v.TextureCube.MipLevels, v.TextureCube.ResourceMinLODClamp);
    if (v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURECUBEARRAY) {
        const auto& c = v.TextureCubeArray;
        return r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && r.Width == r.Height && slices % 6 == 0 && c.NumCubes > 0 &&
               c.First2DArrayFace < slices && u64{c.NumCubes} * 6 <= u64{slices} - c.First2DArrayFace &&
               mip_range(c.MostDetailedMip, c.MipLevels, c.ResourceMinLODClamp);
    }
    if (v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE1D)
        return r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D &&
               mip_range(v.Texture1D.MostDetailedMip, v.Texture1D.MipLevels, v.Texture1D.ResourceMinLODClamp);
    if (v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE1DARRAY) {
        const auto& a = v.Texture1DArray;
        return r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D && mip_range(a.MostDetailedMip, a.MipLevels, a.ResourceMinLODClamp) &&
               a.FirstArraySlice < slices && a.ArraySize > 0 && a.ArraySize <= slices - a.FirstArraySlice;
    }
    if (v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE3D)
        return r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D &&
               mip_range(v.Texture3D.MostDetailedMip, v.Texture3D.MipLevels, v.Texture3D.ResourceMinLODClamp);
    return false;
}
// Which description forms the linear and stand-in paths wrote, one counter each.
enum LinearForm { lf_2d, lf_2d_array, lf_cube, lf_cube_array, lf_1d, lf_1d_array, lf_3d, lf_buffer_srv, lf_buffer_uav, lf_count };
const char* const linear_form_names[lf_count] = {"2d", "2d-array", "cube", "cube-array", "1d", "1d-array", "3d", "buffer-srv", "buffer-uav"};
LONG64 linear_forms[lf_count];
int linear_form(D3D12_SRV_DIMENSION d)
{
    switch (d) {
    case D3D12_SRV_DIMENSION_TEXTURE2DARRAY: return lf_2d_array;
    case D3D12_SRV_DIMENSION_TEXTURECUBE: return lf_cube;
    case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY: return lf_cube_array;
    case D3D12_SRV_DIMENSION_TEXTURE1D: return lf_1d;
    case D3D12_SRV_DIMENSION_TEXTURE1DARRAY: return lf_1d_array;
    case D3D12_SRV_DIMENSION_TEXTURE3D: return lf_3d;
    default: return lf_2d;
    }
}
void note_linear_form(int form, u64 destination)
{
    const LONG64 n = InterlockedIncrement64(&linear_forms[form]);
    if ((n & (n - 1)) == 0)
        log("x2w-memsrv form %s: %lld views written (latest destination 0x%llx)", linear_form_names[form], n, destination);
}
bool linear_memory_srv(const D3D12_RESOURCE_DESC& r, const D3D12_SHADER_RESOURCE_VIEW_DESC& v)
{
    const bool sized = r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        ? r.Width <= D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION && r.Height <= D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION
        : r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D
        ? r.Width <= D3D12_REQ_TEXTURE1D_U_DIMENSION && r.Height == 1 && r.DepthOrArraySize <= D3D12_REQ_TEXTURE1D_ARRAY_AXIS_DIMENSION
        : r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D &&
          r.Width <= D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION && r.Height <= D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION &&
          r.DepthOrArraySize <= D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION;
    return sized && r.Width && r.Height &&
           r.DepthOrArraySize >= 1 && r.MipLevels >= 1 && r.MipLevels <= 15 && r.SampleDesc.Count == 1 &&
           static_cast<UINT>(r.Layout) == 0 && !tiled_bc_srv(r) && !x2w_startup::bc64::supports(r) &&
           !(r.Flags & (D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
                        D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) &&
           v.Format == r.Format && linear_view_in_range(r, v);
}
bool memory_srv_announced_on()
{
    static const bool on = [] {
        const bool v = x2w_local::env_on("X2W_MEMORY_SRV_ANNOUNCED");
        if (v) log("x2w-memsrv X2W_MEMORY_SRV_ANNOUNCED=1: console-swizzled memory SRVs at XG-announced addresses take the linear path");
        return v;
    }();
    return on;
}
bool xg_announced(u64 address)
{
    AcquireSRWLockShared(&written_lock);
    const auto found = written_textures().find(address);
    const bool announced = found != written_textures().end() && !found->second.bc7_width;
    ReleaseSRWLockShared(&written_lock);
    return announced;
}
bool console_swizzle(const D3D12_RESOURCE_DESC& r)
{
    const UINT layout = static_cast<UINT>(r.Layout);
    return layout >= 0x100 && layout <= 0x1ff;
}
bool announced_linear_srv(const D3D12_RESOURCE_DESC& r, const D3D12_SHADER_RESOURCE_VIEW_DESC& v, u64 address)
{
    if (!console_swizzle(r) || tiled_bc_srv(r) || x2w_startup::bc64::supports(r) || !xg_announced(address)) return false;
    D3D12_RESOURCE_DESC as_linear = r;
    as_linear.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    return linear_memory_srv(as_linear, v);
}
std::mutex linear_srv_mutex;
std::unordered_multimap<u64, std::pair<D3D12_RESOURCE_DESC, Wrapper*>> linear_srv_textures;
u64 linear_srv_bytes;  // desktop allocation size of the textures above (GetResourceAllocationInfo)
// detiled: the memory is console-swizzled (X2W_MEMORY_SRV_DETILE); a separate texture from a linear one with the
// same description, so one desktop copy per (address, description, form).
Wrapper* publish_linear_texture(Wrapper* device, u64 address, const D3D12_RESOURCE_DESC& desc, bool detiled = false)
{
    std::lock_guard lock(linear_srv_mutex);
    const auto [first, last] = linear_srv_textures.equal_range(address);
    for (auto it = first; it != last; ++it)
        if (same_texture_description(it->second.first, desc) && it->second.second->own().detiled_memory_srv == detiled)
            return it->second.second;
    const D3D12_RESOURCE_DESC texture = sanitized(desc);
    const D3D12_HEAP_PROPERTIES props{D3D12_HEAP_TYPE_DEFAULT};
    ID3D12Resource* created = nullptr;
    const HRESULT hr = device_of(device)->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &texture, D3D12_RESOURCE_STATE_COMMON,
                                                                  nullptr, IID_PPV_ARGS(&created));
    static LONG shown, failures;
    if (first_calls(shown, 16) || (FAILED(hr) && first_calls(failures, 16)))
        log("x2w-memsrv linear texture at 0x%llx: %llux%ux%u mips %u fmt %d flags 0x%x -> 0x%08lx (%zu so far)", address, desc.Width,
            desc.Height, desc.DepthOrArraySize, desc.MipLevels, desc.Format, static_cast<unsigned>(desc.Flags),
            static_cast<unsigned long>(hr), linear_srv_textures.size() + 1);
    if (FAILED(hr)) return nullptr;
    x2w_startup::mark_packed_stand_in(created, desktop_format(desc.Format));
    Wrapper* w = wrap(created, resource_kind);
    w->own().title_memory = reinterpret_cast<void*>(address);
    w->own().linear_memory_srv = true;
    w->own().detiled_memory_srv = detiled;
    note_resource(w, &desc, 0);
    linear_srv_bytes += device_of(device)->GetResourceAllocationInfo(0, 1, &texture).SizeInBytes;
    linear_srv_textures.emplace(address, std::make_pair(desc, w));
    return w;
}

bool memory_srv_detile_on()
{
    static const bool on = [] {
        const bool v = x2w_local::env_on("X2W_MEMORY_SRV_DETILE");
        if (v) log("x2w-memsrv X2W_MEMORY_SRV_DETILE=1: unannounced console-swizzled memory SRVs are detiled (GFX10 layout)");
        return v;
    }();
    return on;
}
// Bytes per element (per 4x4 block for BC); 0 when the format is not detiled (not a power of two, depth, video).
UINT detile_element_bytes(DXGI_FORMAT f, bool& bc)
{
    bc = true;
    switch (f) {
    case DXGI_FORMAT_BC1_TYPELESS: case DXGI_FORMAT_BC1_UNORM: case DXGI_FORMAT_BC1_UNORM_SRGB:
    case DXGI_FORMAT_BC4_TYPELESS: case DXGI_FORMAT_BC4_UNORM: case DXGI_FORMAT_BC4_SNORM:
        return 8;
    case DXGI_FORMAT_BC2_TYPELESS: case DXGI_FORMAT_BC2_UNORM: case DXGI_FORMAT_BC2_UNORM_SRGB:
    case DXGI_FORMAT_BC3_TYPELESS: case DXGI_FORMAT_BC3_UNORM: case DXGI_FORMAT_BC3_UNORM_SRGB:
    case DXGI_FORMAT_BC5_TYPELESS: case DXGI_FORMAT_BC5_UNORM: case DXGI_FORMAT_BC5_SNORM:
    case DXGI_FORMAT_BC6H_TYPELESS: case DXGI_FORMAT_BC6H_UF16: case DXGI_FORMAT_BC6H_SF16:
    case DXGI_FORMAT_BC7_TYPELESS: case DXGI_FORMAT_BC7_UNORM: case DXGI_FORMAT_BC7_UNORM_SRGB:
        return 16;
    default:
        break;
    }
    bc = false;
    switch (f) {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: return 16;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: case DXGI_FORMAT_R32G32_TYPELESS: return 8;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: case DXGI_FORMAT_R16G16_TYPELESS: case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
        return 4;
    case DXGI_FORMAT_R8G8_TYPELESS: case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_B5G6R5_UNORM:
    case DXGI_FORMAT_B5G5R5A1_UNORM: case DXGI_FORMAT_B4G4R4A4_UNORM:
        return 2;
    case DXGI_FORMAT_R8_TYPELESS: case DXGI_FORMAT_A8_UNORM: return 1;
    default: {
        const UINT n = x2w_startup::typed_buffer_format_bytes(f);
        return n == 12 ? 0 : n;
    }
    }
}
x2w_startup::gfx10_local::Layout detile_layout(const D3D12_RESOURCE_DESC& r)
{
    bool bc = false;
    const UINT bytes = detile_element_bytes(r.Format, bc);
    const UINT layout = static_cast<UINT>(r.Layout);
    // The D modes (0x106, 0x10a) are taken for 1-, 2-, 8- and 16-byte elements (BC blocks are 8 or 16), whose
    // equations addrlib gives the same on Navi1x and Navi21. For 4-byte elements they are left out: there addrlib's
    // equation differs between Navi1x and Navi21 (RB+), and which one the console uses is not known
    // (tests/gfx10_layout_check.cpp --addrlib). Thick D has no equation (gfx10_layout.h), so a 3D D-mode is invalid.
    const bool d_mode = layout == 0x106 || layout == 0x10a;
    const bool known = layout == 0x100 || layout == 0x105 || layout == 0x109 || (d_mode && bytes != 4);
    const bool dim = r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D || r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    if (!bytes || !known || !dim || r.Width > 0xffffffffull)
        return x2w_startup::gfx10_local::Layout(~0u, false, 0, 0, 0, 0, 0);  // invalid
    const UINT w = static_cast<UINT>(r.Width), h = r.Height;
    return x2w_startup::gfx10_local::Layout(layout - 0x100, r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D, bytes,
                                      bc ? (w + 3) / 4 : w, bc ? (h + 3) / 4 : h, r.DepthOrArraySize, r.MipLevels ? r.MipLevels : 1);
}
// Why detile_srv does not take the call, as an index into detile_declines (d3d12_draw.inl); 0 when it does.
// check_memory false (a null address): the description alone, 11 ("would detile") when it passes.
unsigned detile_decline(const D3D12_RESOURCE_DESC& r, const D3D12_SHADER_RESOURCE_VIEW_DESC& v, u64 address, bool check_memory = true)
{
    if (!console_swizzle(r)) return 1;
    if (supports_placed_memory_srv(r, v)) return 2;
    if (check_memory && xg_announced(address)) return 3;
    D3D12_RESOURCE_DESC as_linear = r;
    as_linear.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;  // (tiled_bc_srv / bc64 need a console layout: false here)
    if (!linear_memory_srv(as_linear, v)) return 4;
    const auto l = detile_layout(r);
    if (!l.valid) return 5;
    if (l.size > (1ull << 30)) return 7;
    if (!check_memory) return 11;
    if (!x2w_local::readable(reinterpret_cast<const void*>(address), static_cast<std::size_t>(l.size))) return 8;
    return 0;
}

bool memory_srv_buffer_on()
{
    static const bool on = [] {
        const bool v = x2w_local::env_on("X2W_MEMORY_SRV_BUFFER");
        if (v) log("x2w-memsrv X2W_MEMORY_SRV_BUFFER=1: memory buffer SRVs/UAVs without a buffer resource get a stand-in at the address");
        return v;
    }();
    return on;
}
std::mutex standin_buffer_mutex;
std::unordered_map<u64, std::pair<u64, Wrapper*>> standin_buffers;  // address -> (bytes, wrapper)
Wrapper* standin_memory_buffer(u64 address, const D3D12_RESOURCE_DESC& desc, u64 span)
{
    if (!address || !span || span > (u64{256} << 20) || desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
        !x2w_local::readable(reinterpret_cast<const void*>(address), 1))
        return nullptr;
    const u64 bytes = desc.Width > span ? desc.Width : span;
    std::lock_guard lock(standin_buffer_mutex);
    if (const auto found = standin_buffers.find(address); found != standin_buffers.end() && found->second.first >= bytes)
        return found->second.second;
    D3D12_RESOURCE_DESC console = desc;
    console.Width = bytes;
    Wrapper* w = wrap(nullptr, resource_kind);
    w->own().title_memory = reinterpret_cast<void*>(address);
    note_resource(w, &console, bytes);
    standin_buffers[address] = {bytes, w};
    static LONG shown;
    const std::size_t count = standin_buffers.size();
    if (first_calls(shown, 16) || (count & (count - 1)) == 0)
        log("x2w-memsrv stand-in buffer at 0x%llx: %llu bytes (%zu stand-ins)", address, bytes, count);
    return w;
}

struct MemoryUnorderedAccessView {
    D3D12_UNORDERED_ACCESS_VIEW_DESC view;
    u64 address;
};
static_assert(offsetof(MemoryUnorderedAccessView, address) == 0x28);
void __stdcall device_memory_uav(Wrapper* self, const D3D12_RESOURCE_DESC* resource, const MemoryUnorderedAccessView* input, u64 destination)
{
    static LONG refused;
    const char* reason = nullptr;
    if (!destination) reason = "null_destination";
    else if (!x2w_local::readable(resource, sizeof(*resource))) reason = "unreadable_resource";
    else if (!x2w_local::readable(input, sizeof(*input))) reason = "unreadable_view";
    else if (!input->address) reason = "null_address";
    else if (resource->Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || input->view.ViewDimension != D3D12_UAV_DIMENSION_BUFFER)
        reason = "not_a_buffer";
    else if (!memory_srv_destination(destination)) reason = "destination_outside_view_heaps";
    u64 span = 0;
    if (!reason) {
        const auto& b = input->view.Buffer;
        const bool raw = (b.Flags & D3D12_BUFFER_UAV_FLAG_RAW) != 0;
        const UINT element = b.StructureByteStride ? b.StructureByteStride : raw ? 4 : x2w_startup::typed_buffer_format_bytes(input->view.Format);
        if (b.CounterOffsetInBytes) reason = "counter";
        else if (raw && (input->view.Format != DXGI_FORMAT_R32_TYPELESS || b.StructureByteStride)) reason = "bad_raw_view";
        else if (b.StructureByteStride && input->view.Format != DXGI_FORMAT_UNKNOWN) reason = "bad_structured_view";
        else if (!element || !b.NumElements) reason = "bad_element";
        else if ((span = (b.FirstElement + b.NumElements) * element) > resource->Width) reason = "view_past_resource";
    }
    Wrapper* standin = reason ? nullptr : standin_memory_buffer(input->address, *resource, span);
    if (!reason && !standin) reason = "no_stand_in";
    if (reason) {
        if (first_calls(refused, noop_census_enabled() ? 512 : 8))
            log("x2w-memsrv Device::slot71 memory UAV refused: reason=%s res{dim=%d width=%llu} view{dim=%d fmt=%d} dest=0x%llx", reason,
                x2w_local::readable(resource, sizeof(*resource)) ? resource->Dimension : -1,
                x2w_local::readable(resource, sizeof(*resource)) ? resource->Width : 0ull,
                x2w_local::readable(input, sizeof(*input)) ? input->view.ViewDimension : -1,
                x2w_local::readable(input, sizeof(*input)) ? input->view.Format : -1, destination);
        return;
    }
    note_linear_form(lf_buffer_uav, destination);
    view_slot<19, 3>(self, reinterpret_cast<u64>(standin), 0, reinterpret_cast<u64>(&input->view), destination);
}

enum MemorySrvOutcome { mso_linear, mso_announced, mso_tiled, mso_buffer, mso_standin, mso_refused, mso_detiled, mso_descriptor, mso_count };
const char* const memory_srv_outcome_names[mso_count] = {"linear", "announced-linear", "tiled/bc64", "buffer", "stand-in buffer", "refused", "detiled", "buffer descriptor"};
LONG64 memory_srv_outcomes[mso_count];
std::mutex refusal_census_mutex;
std::unordered_map<u64, LONG64> refusal_census;
LONG64 refusal_census_total;
const char* const census_reasons[] = {"null_destination", "unreadable_resource", "unreadable_view", "null_address",
    "unsupported_description", "destination_outside_view_heaps", "buffer_unavailable", "texture_unavailable"};
bool census_every_event();
bool texture_census_on()
{
    static const bool on = [] {
        const bool v = x2w_local::env_on("X2W_TEXTURE_CENSUS");
        if (v) log("x2w-texcensus X2W_TEXTURE_CENSUS=1: slot-70 outcomes and refusal classes, draw texture bindings real/null; "
                   "all cumulative, every 300 frames%s", census_every_event() ? " and after every event (X2W_TEXTURE_CENSUS_EVERY=1)" : "");
        return v;
    }();
    return on;
}
// X2W_TEXTURE_CENSUS_EVERY=1 (tests): the cumulative lines after every counted event instead of at the thresholds.
bool census_every_event()
{
    static const bool on = x2w_local::env_on("X2W_TEXTURE_CENSUS_EVERY");
    return on;
}
void log_memory_srv_outcomes(const char* when)
{
    LONG64 total = 0;
    for (LONG64 n : memory_srv_outcomes) total += n;
    log("x2w-texcensus slot70 calls %lld (%s, cumulative): linear %lld, announced-linear %lld, tiled/bc64 %lld, buffer %lld, stand-in buffer %lld, "
        "detiled %lld, refused %lld, buffer descriptor %lld", total, when, memory_srv_outcomes[0], memory_srv_outcomes[1], memory_srv_outcomes[2], memory_srv_outcomes[3],
        memory_srv_outcomes[4], memory_srv_outcomes[6], memory_srv_outcomes[5], memory_srv_outcomes[7]);
}
void log_refusal_classes(const char* when)  // refusal_census_mutex held
{
    std::vector<std::pair<LONG64, u64>> sorted;
    for (const auto& [k, n] : refusal_census) sorted.emplace_back(n, k);
    std::sort(sorted.begin(), sorted.end(), std::greater<>());
    log("x2w-texcensus slot70 refusals %lld (%s, cumulative), %zu classes, top classes:", refusal_census_total, when, sorted.size());
    for (std::size_t i = 0; i < sorted.size() && i < 24; ++i) {
        const u64 k = sorted[i].second;
        const u64 decline = (k >> 31) & 15;
        char why[48] = "";
        if (decline) std::snprintf(why, sizeof why, " detile: %s", decline < std::size(detile_declines) ? detile_declines[decline] : "?");
        log("x2w-texcensus   %lld x %s dim %llu layout 0x%llx fmt %llu %s %s %s%s", sorted[i].first,
            (k & 15) < std::size(census_reasons) ? census_reasons[k & 15] : "other", (k >> 4) & 15, (k >> 8) & 0x1ff, (k >> 20) & 0xff,
            (k >> 28) & 1 ? "mips>1" : "mips=1", (k >> 29) & 1 ? "announced" : "not-announced",
            !((k >> 35) & 1) ? "unreadable" : (k >> 30) & 1 ? "nonzero" : "zero", why);
    }
}
// Called with the 300-frame draw census line (d3d12_draw.inl), so a title run's scene window is the difference of
// two reports.
void report_memory_srv_census(long frame)
{
    if (!texture_census_on()) return;
    char when[32];
    std::snprintf(when, sizeof(when), "frame %ld", frame);
    log_memory_srv_outcomes(when);
    std::lock_guard lock(refusal_census_mutex);
    log_refusal_classes(when);
}
void census_memory_srv_outcome(int outcome)
{
    if (!texture_census_on()) return;
    InterlockedIncrement64(&memory_srv_outcomes[outcome]);
    LONG64 total = 0;
    for (LONG64 n : memory_srv_outcomes) total += n;
    if (census_every_event() || (total >= 1024 && (total & (total - 1)) == 0)) log_memory_srv_outcomes("threshold");
}
// The memory at a refused call's address: its first 256 bytes readable, and any of them nonzero.
struct RefusedMemory {
    bool readable = false, nonzero = false;
};
RefusedMemory refused_memory(u64 address)
{
    RefusedMemory m;
    if (address && x2w_local::readable(reinterpret_cast<const void*>(address), 256)) {
        m.readable = true;
        const auto* bytes = reinterpret_cast<const unsigned char*>(address);
        for (int i = 0; i < 256 && !m.nonzero; ++i) m.nonzero = bytes[i] != 0;
    }
    return m;
}
LONG64 detile_decline_counts[std::size(detile_declines)];
// decline: detile_decline's verdict (detile_declines, d3d12_draw.inl; 0 = not asked). Key bits: reason 0-3, dimension
// 4-7, layout 8-16, format 20-27, mips > 1 28, announced 29, nonzero 30, decline 31-34, first 256 bytes readable 35.
// The memory of a refused call, read once for the census and the marker (and not at all when both are off).
RefusedMemory refused_call_memory(const MemoryShaderResourceView* input)
{
    if (!texture_census_on() && !buffer_heap_index_on()) return {};
    return x2w_local::readable(input, sizeof(*input)) ? refused_memory(input->address) : RefusedMemory{};
}
void census_refusal(const char* reason, const D3D12_RESOURCE_DESC* resource, const MemoryShaderResourceView* input, unsigned decline,
                    const RefusedMemory& memory)
{
    if (!texture_census_on()) return;
    census_memory_srv_outcome(mso_refused);
    u64 r = 0;
    while (r < std::size(census_reasons) && std::strcmp(census_reasons[r], reason) != 0) ++r;
    const bool resource_ok = x2w_local::readable(resource, sizeof(*resource));
    const bool input_ok = x2w_local::readable(input, sizeof(*input));
    const u64 address = input_ok ? input->address : 0;
    if (decline >= std::size(detile_declines)) decline = 0;
    const u64 dim = resource_ok && resource->Dimension >= 0 && resource->Dimension <= 4 ? resource->Dimension : 7;
    const u64 layout = resource_ok ? (static_cast<UINT>(resource->Layout) & 0x1ff) : 0x1ff;
    const u64 format = resource_ok ? (static_cast<UINT>(resource->Format) & 0xff) : 0xff;
    const u64 chain = resource_ok && resource->MipLevels > 1;
    const u64 key = r | dim << 4 | layout << 8 | format << 20 | chain << 28 | u64{address && xg_announced(address)} << 29 |
                    u64{memory.nonzero} << 30 | u64{decline} << 31 | u64{memory.readable} << 35;
    if (decline) {
        InterlockedIncrement64(&detile_decline_counts[decline]);
        LONG64 n = 0;
        for (LONG64 c : detile_decline_counts) n += c;
        if (census_every_event() || (n & (n - 1)) == 0) {
            char parts[512];
            int used = 0;
            for (std::size_t i = 1; i < std::size(detile_declines); ++i)
                used += std::snprintf(parts + used, sizeof(parts) - used, "%s%s %lld", i > 1 ? ", " : "", detile_declines[i], detile_decline_counts[i]);
            log("x2w-texcensus detile declines of refused texture calls %lld (X2W_MEMORY_SRV_DETILE): %s", n, parts);
        }
    }
    std::lock_guard lock(refusal_census_mutex);
    const bool fresh = ++refusal_census[key] == 1;
    const LONG64 total = ++refusal_census_total;
    // The first call of each class (at most 96 classes): every field, the layout's size and how much of it is readable.
    static LONG described_classes;
    if (fresh && resource_ok && input_ok && first_calls(described_classes, 96)) {
        char resource_text[224], view_text[256];
        x2w_startup::report_resource(resource_text, sizeof(resource_text), resource);
        x2w_startup::report_view(view_text, sizeof(view_text), &input->view);
        const auto l = detile_layout(*resource);
        const std::size_t want = l.valid ? static_cast<std::size_t>((std::min)(l.size, u64{1} << 30)) : 0;
        log("x2w-texcensus new refusal class: %s res{%s} view{%s} address 0x%llx; detile: %s; memory %s; console layout %llu bytes, %zu "
            "readable from the address", reason, resource_text, view_text, address, detile_declines[decline],
            !memory.readable ? "unreadable" : memory.nonzero ? "nonzero" : "zero", l.valid ? l.size : 0ull,
            want && address ? readable_window(address, want) : 0);
    }
    if (census_every_event() || (total >= 256 && (total & (total - 1)) == 0)) log_refusal_classes("threshold");
}

u64 __stdcall device_slot61_census(Wrapper* self, u64 rdx, u64 r8, u64 r9)
{
    if (noop_census_enabled()) {  // what the no-op it replaces still reports
        const u64 args[8] = {reinterpret_cast<u64>(self), rdx, r8, r9, 0, 0, 0, 0};
        noop_census_note(device_kind, 61, args, reinterpret_cast<u64>(_ReturnAddress()) - reinterpret_cast<u64>(GetModuleHandleW(nullptr)));
    }
    static LONG calls;
    if (first_calls(calls, 16)) {
        char hex[200] = "unreadable";
        if (x2w_local::readable(reinterpret_cast<const void*>(rdx), 64)) {
            const auto* w = reinterpret_cast<const std::uint32_t*>(rdx);
            int used = 0;
            for (int i = 0; i < 16; ++i) used += std::snprintf(hex + used, sizeof(hex) - used, "%08x ", w[i]);
        }
        log("x2w-texcensus Device::slot61 (sampler?) rdx=0x%llx r8=0x%llx r9=0x%llx words %s", rdx, r8, r9, hex);
    }
    return 0;
}

// The first check that refuses a call, in the order the checks run; null when every check passes.
const char* memory_srv_refusal(const D3D12_RESOURCE_DESC* resource, const MemoryShaderResourceView* input, u64 destination)
{
    if (!destination) return "null_destination";
    const bool readable_input = x2w_local::readable(input, sizeof(*input));
    // An acceleration-structure view names its structure by address only; the resource description is not read.
    const bool as_view = readable_input && x2w_startup::is_as_srv(input->view);
    if (!as_view && !x2w_local::readable(resource, sizeof(*resource))) return "unreadable_resource";
    if (!readable_input) return "unreadable_view";
    if (as_view) {
        if (!x2w_startup::as_srv_location(input->view, input->address)) return "null_address";
        if (!x2w_startup::supports_as_memory_srv(input->view)) return "unsupported_description";
    } else {
        if (!input->address) return "null_address";
        if (!supports_memory_srv(*resource, input->view)) return "unsupported_description";
    }
    if (!memory_srv_destination(destination)) return "destination_outside_view_heaps";
    return nullptr;
}

// One line with every field the checks read; the caller rate-limits it.
void log_refused_memory_srv(const char* what, const char* reason, const D3D12_RESOURCE_DESC* resource,
                            const MemoryShaderResourceView* input, u64 destination)
{
    const bool resource_ok = resource && x2w_local::readable(resource, sizeof(*resource));
    const bool input_ok = x2w_local::readable(input, sizeof(*input));
    const u64 address = input_ok ? input->address : 0;
    char line[640];
    x2w_startup::report_refused_memory_srv(line, sizeof(line), what, reason, resource_ok ? resource : nullptr,
        input_ok ? &input->view : nullptr, address, address && x2w_local::readable(reinterpret_cast<const void*>(address), 1),
        destination, destination && memory_srv_destination(destination));
    log("%s", line);
}

void mark_refused_destination(u64 destination, const char* reason, const D3D12_RESOURCE_DESC* resource,
                              const MemoryShaderResourceView* input, unsigned decline, const RefusedMemory& memory)
{
    if (!buffer_heap_index_on() || !destination || !title_backed(destination, 32) || !x2w_local::writable(reinterpret_cast<void*>(destination), 32))
        return;
    std::uint32_t marker[8] = {refused_view_magic, refusal_code(reason), 1u << 31, 0, 0, 0, 0, 0};
    if (x2w_local::readable(resource, sizeof(*resource))) {
        const auto& r = *resource;
        marker[2] |= (static_cast<UINT>(r.Format) & 0xff) | (static_cast<UINT>(r.Layout) & 0x1ff) << 8 |
                     (static_cast<UINT>(r.Dimension) & 7) << 17 | (r.MipLevels < 15 ? r.MipLevels : 15u) << 20 | (decline & 15) << 24;
        marker[3] = r.Width > 0xffffffffull ? 0xffffffffu : static_cast<std::uint32_t>(r.Width);
        marker[4] = r.Height;
        marker[5] = r.DepthOrArraySize;
    }
    if (x2w_local::readable(input, sizeof(*input))) {
        marker[5] |= std::uint32_t{memory.readable} << 16 | std::uint32_t{memory.nonzero} << 17;
        marker[6] = static_cast<std::uint32_t>(input->address);
        marker[7] = static_cast<std::uint32_t>(input->address >> 32);
    }
    std::memcpy(reinterpret_cast<void*>(destination), marker, sizeof(marker));
    if (const u64 desktop = desktop_cpu_handle(destination)) {
        AcquireSRWLockExclusive(&note_lock);
        if (std::uint32_t* slot = shadow_slot(desktop)) *slot = 0;
        ReleaseSRWLockExclusive(&note_lock);
    }
}
LONG64 vdesc_no_buffer, vdesc_unaligned, vdesc_no_destination, vdesc_layout, vdesc_ticks[3], vdesc_lock_taken;
LONG64 qpc_now()
{
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart;
}
bool write_buffer_descriptor(u64 destination, const MemoryShaderResourceView* input, bool found_buffer, LONG64 lookup_ticks)
{
    if (!buffer_heap_index_on()) return false;
    const auto counted = [](LONG64& which) {
        InterlockedIncrement64(&which);
        const LONG64 n = vdesc_no_buffer + vdesc_unaligned + vdesc_no_destination + vdesc_layout;
        if ((n & (n - 1)) == 0) {
            static const double us_per_tick = [] {
                LARGE_INTEGER f;
                QueryPerformanceFrequency(&f);
                return 1e6 / static_cast<double>(f.QuadPart);
            }();
            const double written = static_cast<double>((std::max)(1ll, vdesc_no_buffer + vdesc_unaligned));
            log("x2w-memsrv buffer descriptors: written %lld (no buffer %lld, buffer at an offset the view cannot name %lld), not written: "
                "destination not title memory %lld, view not encodable %lld; per written call: buffer lookup %.2f us, destination checks "
                "%.2f us, shadow clear %.2f us (lock taken %lld)", vdesc_no_buffer + vdesc_unaligned, vdesc_no_buffer, vdesc_unaligned,
                vdesc_no_destination, vdesc_layout, vdesc_ticks[0] * us_per_tick / written, vdesc_ticks[1] * us_per_tick / written,
                vdesc_ticks[2] * us_per_tick / written, vdesc_lock_taken);
        }
    };
    const LONG64 t0 = qpc_now();
    if (!destination || !title_backed(destination, 32) || !x2w_local::writable(reinterpret_cast<void*>(destination), 32)) {
        counted(vdesc_no_destination);
        return false;
    }
    const LONG64 t1 = qpc_now();
    x2w_startup::BufferViewLayout l;
    if (!x2w_startup::buffer_view_layout_from_desc(false, reinterpret_cast<const unsigned char*>(&input->view), 40, l) ||
        l.kind == x2w_startup::BufferViewKind::unknown || !l.num_records || ((input->address + l.offset) >> 48)) {
        counted(vdesc_layout);
        return false;
    }
    std::uint32_t words[8];
    x2w_startup::gfx10_buffer_words(l, words);
    const u64 base = input->address + l.offset;
    words[0] = static_cast<std::uint32_t>(base);
    words[1] |= static_cast<std::uint32_t>(base >> 32) & 0xffff;
    std::memcpy(reinterpret_cast<void*>(destination), words, sizeof(words));
    if (const u64 desktop = desktop_cpu_handle(destination)) {
        // Most slots hold no id (the title rewrites the same slots with V#s): look under the shared lock first.
        if (shadow_id(desktop)) {
            InterlockedIncrement64(&vdesc_lock_taken);
            AcquireSRWLockExclusive(&note_lock);
            if (std::uint32_t* slot = shadow_slot(desktop)) *slot = 0;
            ReleaseSRWLockExclusive(&note_lock);
        }
    }
    const LONG64 t2 = qpc_now();
    InterlockedAdd64(&vdesc_ticks[0], lookup_ticks);
    InterlockedAdd64(&vdesc_ticks[1], t1 - t0);
    InterlockedAdd64(&vdesc_ticks[2], t2 - t1);
    counted(found_buffer ? vdesc_unaligned : vdesc_no_buffer);
    static LONG shown;
    if (first_calls(shown, 8))
        log("x2w-memsrv buffer descriptor written at 0x%llx (X2W_BUFFER_HEAP_INDEX): base 0x%llx, stride %u, records %u, word 3 0x%08x%s", destination,
            base, l.index_stride, l.num_records, words[3], found_buffer ? " (a buffer is there, at an offset the view cannot name)" : "");
    return true;
}
// Counts a refusal (memory_srv_census) and logs its line while the first 8 refusals last, and also for
// the first refusal of each of the first few distinct shapes (MemorySrvCensus::detail_shapes).
void refuse_memory_srv(LONG& refused, const char* what, const char* reason, const D3D12_RESOURCE_DESC* resource,
                       const MemoryShaderResourceView* input, u64 destination)
{
    const bool within_limit = first_calls(refused, noop_census_enabled() ? 2048 : 8);
    const bool new_shape = memory_srv_census.note(reason, x2w_local::readable(resource, sizeof(*resource)) ? resource : nullptr);
    if (within_limit || new_shape) log_refused_memory_srv(what, reason, resource, input, destination);
}

// Counts a successful write (memory_srv_writes) and logs the first write to each of the first few
// distinct destination descriptors.
void note_memory_srv_write(const D3D12_RESOURCE_DESC* resource, const MemoryShaderResourceView* input, u64 destination)
{
    const u64 ordinal = memory_srv_writes.note(destination);
    if (!ordinal) return;
    char resource_text[224], view_text[256];
    x2w_startup::report_resource(resource_text, sizeof(resource_text), resource);
    x2w_startup::report_view(view_text, sizeof(view_text), &input->view);
    log("x2w-d3d12 memory SRV written %llu: dest=0x%llx address=0x%llx res{%s} view{%s}", ordinal, destination, input->address,
        resource_text, view_text);
}

// The placed buffer that holds `address`, as the pure resolver wants it; nothing found when no live one does.
x2w_startup::PlacedAsBuffer placed_as_buffer(const Wrapper* device, u64 address)
{
    x2w_startup::PlacedAsBuffer result;
    const auto placed = placed_buffer_range(device, address, 1);
    if (placed.reject) {
        result.ambiguous = true;
        return result;
    }
    if (!placed.owner || InterlockedCompareExchange(&placed.owner->own().refs, 0, 0) <= 0) return result;
    const auto* info = resource_info(placed.owner);
    auto* real = static_cast<ID3D12Resource*>(placed.owner->own().real);
    if (!info || !real) return result;
    result.found = true;
    result.title_base = reinterpret_cast<u64>(placed.owner->own().title_memory);
    result.gpu_base = real->GetGPUVirtualAddress();
    result.bytes = info->bytes;
    result.initial_state = info->initial_state;
    return result;
}

// Slot 70 with an acceleration-structure view: published through slot 18 with no resource and the
// desktop address; a refusal leaves the destination descriptor as it was.
void publish_as_srv(Wrapper* self, const D3D12_RESOURCE_DESC* resource, const MemoryShaderResourceView* input, u64 destination)
{
    static LONG refused;
    const u64 location = x2w_startup::as_srv_location(input->view, input->address);
    D3D12_SHADER_RESOURCE_VIEW_DESC resolved{};
    const auto refusal = x2w_startup::resolve_as_srv_desc(location, placed_as_buffer(self, location), input->view, resolved);
    if (refusal != x2w_startup::AsSrvRefusal::none) {
        refuse_memory_srv(refused, "unavailable acceleration structure at the supplied address",
                          x2w_startup::as_srv_refusal_name(refusal), resource, input, destination);
        return;
    }
    view_slot<18, 2>(self, 0, reinterpret_cast<u64>(&resolved), destination, 0);
}

void __stdcall device_memory_srv(Wrapper* self, const D3D12_RESOURCE_DESC* resource,
                                 const MemoryShaderResourceView* input, u64 destination)
{
    X2W_ZONE("device_memory_srv");
    static LONG refused;
    check_resource_address_index();
    if (noop_census_enabled()) {
        static LONG64 calls;
        const LONG64 n = InterlockedIncrement64(&calls);
        CensusHeap h{};
        const bool in_heap = noop_census_heap_index(destination, &h);
        if (n <= 8 || (n & (n - 1)) == 0)
            log("x2w-noop-census memory SRV call %lld: destination 0x%llx = %s index 0x%llx of the heap at 0x%llx", n, destination,
                in_heap ? (h.desktop ? "DESKTOP" : "title") : "no heap", h.index, h.base);
    }
    const bool described = destination && x2w_local::readable(resource, sizeof(*resource)) &&
        x2w_local::readable(input, sizeof(*input)) && input->address;
    const bool plain = described && memory_srv_linear_on() && linear_memory_srv(*resource, input->view);
    if ((plain || (described && memory_srv_announced_on() && announced_linear_srv(*resource, input->view, input->address))) &&
        memory_srv_destination(destination)) {
        Wrapper* texture = existing_memory_texture(input->address, *resource);
        if (!texture) texture = publish_linear_texture(self, input->address, *resource);
        if (texture) {
            static LONG64 linear_views;
            const LONG64 n = InterlockedIncrement64(&linear_views);
            if ((n & (n - 1)) == 0) {
                std::size_t textures;
                u64 bytes;
                {
                    std::lock_guard lock(linear_srv_mutex);
                    textures = linear_srv_textures.size();
                    bytes = linear_srv_bytes;
                }
                log("x2w-memsrv linear memory SRVs written: %lld, %zu textures, %llu KiB (latest destination 0x%llx)", n, textures,
                    bytes >> 10, destination);
            }
            note_linear_form(linear_form(input->view.ViewDimension), destination);
            census_memory_srv_outcome(plain ? mso_linear : mso_announced);
            view_slot<18, 2>(self, reinterpret_cast<u64>(texture), reinterpret_cast<u64>(&input->view), destination, 0);
            return;
        }
    }
    unsigned decline = 0;
    if (memory_srv_detile_on() && destination && x2w_local::readable(resource, sizeof(*resource)) &&
        x2w_local::readable(input, sizeof(*input)) && resource->Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) {
        decline = detile_decline(*resource, input->view, input->address, input->address != 0);
        if (!decline && !memory_srv_destination(destination)) decline = 9;
    }
    if (described && memory_srv_detile_on() && resource->Dimension != D3D12_RESOURCE_DIMENSION_BUFFER && !decline) {
        Wrapper* texture = existing_memory_texture(input->address, *resource);
        if (!texture) texture = publish_linear_texture(self, input->address, *resource, true);
        if (!texture) decline = 10;
        if (texture) {
            static LONG shown;
            if (first_calls(shown, 16))
                log("x2w-memsrv detiled memory SRV at 0x%llx: %llux%ux%u mips %u fmt %d layout 0x%x dim %d", input->address,
                    resource->Width, resource->Height, resource->DepthOrArraySize, resource->MipLevels, resource->Format,
                    static_cast<unsigned>(resource->Layout), resource->Dimension);
            note_linear_form(linear_form(input->view.ViewDimension), destination);
            census_memory_srv_outcome(mso_detiled);
            view_slot<18, 2>(self, reinterpret_cast<u64>(texture), reinterpret_cast<u64>(&input->view), destination, 0);
            return;
        }
    }
    if (const char* reason = memory_srv_refusal(resource, input, destination)) {
        refuse_memory_srv(refused, "unsupported memory SRV arguments", reason, resource, input, destination);
        const RefusedMemory memory = refused_call_memory(input);
        census_refusal(reason, resource, input, decline, memory);
        mark_refused_destination(destination, reason, resource, input, decline, memory);
        return;
    }
    // A single-element texture is published by address like a tiled view: its texel is read at offset 0.
    if (upstream_console_srv_on() && x2w_startup::supports_single_element_srv(*resource, input->view)) {
        if (gpu_written_overlap(input->address, x2w_startup::single_element_bytes(*resource))) {
            refuse_memory_srv(refused, "GPU-written resource at the supplied address", "gpu_written_resource", resource, input,
                              destination);
        } else if (Wrapper* texture = publish_tiled_texture(self, input->address, *resource)) {
            view_slot<18, 2>(self, reinterpret_cast<u64>(texture), reinterpret_cast<u64>(&input->view), destination, 0);
            note_memory_srv_write(resource, input, destination);
        } else {
            refuse_memory_srv(refused, "unavailable texture at the supplied base and layout", "texture_unavailable", resource, input,
                              destination);
        }
        return;
    }
    if (x2w_startup::is_as_srv(input->view)) {
        publish_as_srv(self, resource, input, destination);
        return;
    }
    if (resource->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
        const u64 elem_bytes = x2w_startup::buffer_srv_element_bytes(input->view);
        const u64 span_bytes = (input->view.Buffer.FirstElement + input->view.Buffer.NumElements) * elem_bytes;
        const LONG64 lookup_start = buffer_heap_index_on() ? qpc_now() : 0;
        Wrapper* buffer = existing_memory_buffer(self, input->address, span_bytes);
        const LONG64 lookup_ticks = buffer_heap_index_on() ? qpc_now() - lookup_start : 0;
        const ResourceInfo* info = buffer ? resource_info(buffer) : nullptr;
        D3D12_SHADER_RESOURCE_VIEW_DESC adjusted{};
        if ((!buffer || !info) && memory_srv_buffer_on()) {
            if (Wrapper* standin = standin_memory_buffer(input->address, *resource, span_bytes)) {
                note_linear_form(lf_buffer_srv, destination);
                census_memory_srv_outcome(mso_standin);
                view_slot<18, 2>(self, reinterpret_cast<u64>(standin), reinterpret_cast<u64>(&input->view), destination, 0);
                return;
            }
        }
        if (!buffer || !info ||
            !x2w_startup::resolve_buffer_srv_desc(reinterpret_cast<u64>(buffer->own().title_memory),
                                                  info->console.Width, input->address, input->view, adjusted)) {
            if (write_buffer_descriptor(destination, input, buffer && info, lookup_ticks)) {
                census_memory_srv_outcome(mso_descriptor);
                return;
            }
            refuse_memory_srv(refused, "unavailable buffer at the supplied address", "buffer_unavailable", resource, input,
                              destination);
            const RefusedMemory memory = refused_call_memory(input);
            census_refusal("buffer_unavailable", resource, input, 0, memory);
            mark_refused_destination(destination, "buffer_unavailable", resource, input, 0, memory);
            return;
        }
        census_memory_srv_outcome(mso_buffer);
        view_slot<18, 2>(self, reinterpret_cast<u64>(buffer), reinterpret_cast<u64>(&adjusted), destination, 0);
        note_memory_srv_write(resource, input, destination);
        return;
    }
    // A partial-tile view stays refused when the last XG stand-in copy to the address wrote desktop-footprint
    // rows, or when its padded tiles run past the size the stand-in reported, which the title may not own.
    bool first = false;
    const char* held = nullptr;
    if (partial_tiled_bc(*resource)) {
        {
            std::lock_guard lock(tiled_srv_mutex);
            first = !tiled_srv_addresses.count(input->address);
        }
        if (latest_texture_write_is_rows(input->address)) held = "partial_tile_xg_rows";
        else if (x2w_startup::Bc16Surface(*resource).size() > xg_reported_bytes(self, *resource))
            held = "partial_tile_beyond_xg_size";
    }
    Wrapper* texture = held ? nullptr
        : (tiled_bc_srv(*resource) || x2w_startup::bc64::supports(*resource)) ? publish_tiled_texture(self, input->address, *resource)
                                              : existing_memory_texture(input->address, *resource);
    if (first) log_partial_tile_view(*resource, input->address, held, texture != nullptr);
    // A new console-layout view reads the address only when there is no matching live resource.
    const char* outcome = tiled_bc_srv(*resource) || x2w_startup::bc64::supports(*resource) ? "published BC shape" : "existing resource";
    if (console_layout_srv(*resource)) {
        if ((!texture || !supports_placed_memory_srv(*resource, input->view)) &&
            !desktop_address_view_supported(self, *resource, input->view)) {
            held = "unsupported_desktop_format";
            texture = nullptr;
        } else if (!texture) {
            held = console_address_hold(self, input->address, *resource);
            if (!held && address_resource_overlap(input->address, address_texture_bytes(self, *resource)))
                held = "overlapping_resource";
            texture = held ? nullptr : publish_tiled_texture(self, input->address, *resource);
            outcome = "published console layout";
        }
    }
    if (!texture) outcome = held ? held : "texture_unavailable (no resource there, or texels not readable or over 64 MiB)";
    log_address_kind(outcome, *resource, input->view, input->address);
    if (!texture) {
        refuse_memory_srv(refused, "unavailable texture at the supplied base and layout", held ? held : "texture_unavailable",
                          resource, input, destination);
        const RefusedMemory memory = refused_call_memory(input);
        census_refusal("texture_unavailable", resource, input, decline, memory);
        mark_refused_destination(destination, "texture_unavailable", resource, input, decline, memory);
        return;
    }
    census_memory_srv_outcome(mso_tiled);
    view_slot<18, 2>(self, reinterpret_cast<u64>(texture), reinterpret_cast<u64>(&input->view), destination, 0);
    note_memory_srv_write(resource, input, destination);
}
