// Translated compute programs at Dispatch (list slot 14). A device slot 67 pipeline is a placeholder that keeps
// the key of its published request; gfx2hlsl.py --pending answers the request with <key>.txt (`stage cs`) and
// <key>.cs.cso. Bound from the compute tables recorded by slot 35: raw buffers (`srv <i> buf table <param> <byte>`
// inputs, `uav <i> table <param> <byte>` outputs) named by hardware buffer descriptors, and 2D textures
// (`srv <i> tex2d table <param> <byte>` inputs, `uav <i> tex2d float table <param> <byte> channels 4` outputs) named
// by the token of a view the title created (device slots 18 and 19). `smp` lines become static samplers. Entry
// values come from the compute root constants of slots 37 and 39. Any other manifest line makes the translation
// unavailable; a dispatch that cannot be bound completely is refused with one reason and records no Dispatch and
// binds nothing (the upload of an input texture the title filled in memory may already be recorded: it only
// syncs content).

struct ComputeTranslation {
    // Root parameter and byte offset (views), or parameter and word (entry values). A heap view names the root
    // constant (parameter and word) that indexes its 32-byte record in the list's view heap.
    struct Slot { int a, b; bool texture, heap; };
    u64 key = 0;
    int state = 0;                    // 0 not loaded, 1 ready, 2 unavailable
    ULONGLONG retry_manifest_at = 0;  // only an absent manifest can become ready later
    std::vector<Slot> uds, srvs, uavs;
    x2w_startup::TranslatedSamplers samplers;  // static ones only
    x2w_startup::ComPtr<ID3D12RootSignature> root;
    x2w_startup::ComPtr<ID3D12PipelineState> pipeline;
    const char* unmodelled = nullptr;      // why a manifest that exists cannot be used
    u64 dispatches = 0, translated = 0;    // census: Dispatch calls with this key bound, and those recorded
    const char* first_refusal = nullptr;   // reason of the first refused call
    std::uint32_t refusal_words[4] = {};   // the table descriptor that reason was about, if any; a view token's
                                           // last word is replaced by its view format | dimension << 16
    ~ComputeTranslation()
    {
        // The per-process cache is destroyed during DLL detach. A driver may already have shut
        // down when the process is terminating. Explicit unload still releases both references.
        if (x2w_startup::d3d12_process_terminating) {
            pipeline.Detach();
            root.Detach();
        }
    }
};

// Formats a `float` typed output may be viewed with: the program stores float4.
bool float_output_format(DXGI_FORMAT format)
{
    switch (format) {
    case DXGI_FORMAT_R32G32B32A32_FLOAT: case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_SNORM: case DXGI_FORMAT_R32G32_FLOAT: case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R11G11B10_FLOAT: case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_SNORM:
    case DXGI_FORMAT_R16G16_FLOAT: case DXGI_FORMAT_R16G16_UNORM: case DXGI_FORMAT_R16G16_SNORM: case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R8G8_UNORM: case DXGI_FORMAT_R8G8_SNORM: case DXGI_FORMAT_R16_FLOAT: case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R16_SNORM: case DXGI_FORMAT_R8_UNORM: case DXGI_FORMAT_R8_SNORM: case DXGI_FORMAT_B8G8R8A8_UNORM:
        return true;
    default:
        return false;
    }
}
std::unordered_map<u64, ComputeTranslation>& compute_translations() { static std::unordered_map<u64, ComputeTranslation> m; return m; }
LONG compute_dispatches_translated, compute_dispatches_refused, compute_dispatches_untranslated, compute_dispatches_unmodelled;

// draw_lock held. Returns the reason the translation is unavailable, or null when it is ready or still absent.
const char* load_compute_translation(ComputeTranslation& tr)
{
    tr.state = 2;
    wchar_t path[MAX_PATH];
    std::string manifest;
    if (!cache_path(path, L"", tr.key, L".txt") || !read_text(path, manifest)) {
        tr.retry_manifest_at = GetTickCount64() + 1000;
        return nullptr;
    }
    tr.retry_manifest_at = 0;
    unsigned declared_uds = 0, line_number = 0;
    for (std::size_t at = 0; at < manifest.size(); ++line_number) {
        std::size_t end = manifest.find('\n', at);
        if (end == std::string::npos) end = manifest.size();
        std::string line = manifest.substr(at, end - at);
        at = end + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line_number == 0) {
            if (line != "x2w-manifest 2") return "manifest version";
            continue;
        }
        if (line_number == 1) {
            if (line != "stage cs") return "not a compute manifest";
            continue;
        }
        unsigned i = 0, x = 0, y = 0, z = 0, w = 0;
        int a = 0, b = 0, used = 0;
        const auto whole = [&] { return used == static_cast<int>(line.size()); };
        if (line.empty() || (std::sscanf(line.c_str(), "group %u %u %u%n", &x, &y, &z, &used) == 3 && whole())) continue;
        if (std::sscanf(line.c_str(), "ud %u%n", &declared_uds, &used) == 1 && whole()) continue;
        if (std::sscanf(line.c_str(), "udsrc %u rootconst %d %d%n", &i, &a, &b, &used) == 3 && whole() && i == tr.uds.size())
            tr.uds.push_back({a, b});
        else if (std::sscanf(line.c_str(), "srv %u buf table %d %d%n", &i, &a, &b, &used) == 3 && whole() && i == tr.srvs.size())
            tr.srvs.push_back({a, b});
        else if (std::sscanf(line.c_str(), "srv %u buf heap rootconst %d %d%n", &i, &a, &b, &used) == 3 && whole() &&
                 i == tr.srvs.size())
            tr.srvs.push_back({a, b, false, true});
        else if (std::sscanf(line.c_str(), "uav %u table %d %d%n", &i, &a, &b, &used) == 3 && whole() && i == tr.uavs.size())
            tr.uavs.push_back({a, b});
        else if (std::sscanf(line.c_str(), "uav %u tex2d float table %d %d channels 4%n", &i, &a, &b, &used) == 3 && whole() &&
                 i == tr.uavs.size())
            tr.uavs.push_back({a, b, true});
        else if (std::sscanf(line.c_str(), "srv %u tex2d table %d %d%n", &i, &a, &b, &used) == 3 && whole() && i == tr.srvs.size())
            tr.srvs.push_back({a, b, true});
        else if (std::sscanf(line.c_str(), "smp %u %x %x %x %x%n", &i, &x, &y, &z, &w, &used) == 5 && whole() &&
                 i == tr.samplers.statics.size())
            tr.samplers.read_manifest_line(line.c_str());
        else return line.rfind("srv", 0) == 0 ? "a resource that is not a raw table buffer or a 2D texture"
                  : line.rfind("uav", 0) == 0 ? "an output that is not a raw table buffer or a 2D float texture"
                  : line.rfind("udsrc", 0) == 0 ? "an entry value that is not a root constant"
                  : line.rfind("smp", 0) == 0 ? "a sampler that is not a plain static one" : "an unknown manifest line";
    }
    if (tr.uds.size() != declared_uds || tr.uds.size() > 32 || tr.srvs.size() > 32 || tr.uavs.size() > 8)
        return "entry value, input or output count";
    x2w_startup::ComPtr<ID3DBlob> code;
    if (!cache_path(path, L"", tr.key, L".cs.cso") || FAILED(D3DReadFileToBlob(path, &code))) return "no compiled program";
    // Parameter 0 keeps one constant for a program without entry values: a constants parameter cannot be empty.
    if (FAILED(x2w_startup::create_translated_root(device_of(the_device), static_cast<UINT>(tr.uds.empty() ? 1 : tr.uds.size()),
            static_cast<UINT>(tr.srvs.size()), tr.samplers, &tr.root, static_cast<UINT>(tr.uavs.size()))))
        return "root signature creation failed";
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = tr.root.Get();
    desc.CS = {code->GetBufferPointer(), code->GetBufferSize()};
    if (FAILED(device_of(the_device)->CreateComputePipelineState(&desc, IID_PPV_ARGS(&tr.pipeline))))
        return "pipeline creation failed";
    tr.state = 1;
    log("x2w-draw compute pipeline %016llx translated: %zu entry values, %zu inputs, %zu outputs", tr.key, tr.uds.size(),
        tr.srvs.size(), tr.uavs.size());
    return nullptr;
}

// True when the bound pipeline is a slot 67 placeholder with a request key: the call is then settled here
// (dispatched, or refused and logged). False leaves it to the desktop-pipeline decision of list_dispatch.
bool dispatch_translated(Wrapper* self, UINT x, UINT y, UINT z)
{
    auto* list = recording(self);
    Pass& pass = pass_of(self);
    const u64 key = list && pass.pipeline ? pass.pipeline->own().compute_request_key : 0;
    if (!key) return false;
    const char* reason = nullptr;
    LONG* counter = &compute_dispatches_refused;
    struct View {
        ID3D12Resource* resource;
        u64 offset, bytes;
        bool transition, texture, described;
        unsigned char desc[40];  // a texture view's description, as the title gave it with a desktop format
        Wrapper* wrapper;
    } views[40]{};
    static_assert(sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC) == 40 && sizeof(D3D12_UNORDERED_ACCESS_VIEW_DESC) == 40);
    const auto type = list->GetType();
    AcquireSRWLockExclusive(&draw_lock);
    ComputeTranslation* const tr = &compute_translations()[key];
    tr->key = key;
    ++tr->dispatches;
    std::uint32_t words[8] = {};
    if (type != D3D12_COMMAND_LIST_TYPE_DIRECT && type != D3D12_COMMAND_LIST_TYPE_COMPUTE) reason = "the list type has no Dispatch";
    else if (x > x2w_startup::max_dispatch_groups || y > x2w_startup::max_dispatch_groups || z > x2w_startup::max_dispatch_groups)
        reason = "a group count is over the limit";
    else if (pass.compute_root_wrapper != pass.pipeline->own().compute_request_root)
        reason = "the compute root set is not the one the pipeline was created with";
    else if (!ensure_draw_heap(self->own().list_device)) reason = "no descriptor heap on this device";
    else {
        if (!tr->state || (tr->retry_manifest_at && GetTickCount64() >= tr->retry_manifest_at)) {
            tr->unmodelled = load_compute_translation(*tr);
            if (tr->unmodelled) log("x2w-draw compute pipeline %016llx not translated: %s", key, tr->unmodelled);
        }
        if (tr->state != 1) {
            // A manifest the runtime cannot bind is the dispatch side's work; an absent one is the translator's.
            reason = tr->unmodelled ? tr->unmodelled : "no translation";
            counter = tr->unmodelled ? &compute_dispatches_unmodelled : &compute_dispatches_untranslated;
        }
    }
    // A word no slot 37 or 39 call wrote since the compute root was set has no value to pass (it is not 0).
    for (std::size_t i = 0; !reason && i < tr->uds.size(); ++i)
        if (!pass.compute_constants.is_bound(tr->uds[i].a, tr->uds[i].b, 1)) reason = "an entry value was not set since the compute root";
    const std::size_t inputs = reason ? 0 : tr->srvs.size(), count = reason ? 0 : inputs + tr->uavs.size();
    for (std::size_t i = 0; i < count && !reason; ++i) {
        const bool output = i >= inputs;
        const ComputeTranslation::Slot slot = output ? tr->uavs[i - inputs] : tr->srvs[i];
        // A heap record sits at the 32-bit byte offset the program computes, the root constant word shifted left by
        // 5, in the title memory of the list's view heap: a literal byte offset, as a table's is.
        const u64 entry = slot.a < 0 || slot.b < 0 ? 0
            : slot.heap ? pass.title_view_heap.record(std::uint32_t{pass.compute_constants.word(slot.a, slot.b)} << 5, 16)
            : pass.compute_tables.table(slot.a).record(static_cast<u64>(slot.b), slot.texture ? 32 : 16);
        std::memset(words, 0, sizeof(words));
        const std::size_t entry_bytes = slot.texture ? 32 : 16;
        if (!entry || !x2w_local::readable(reinterpret_cast<const void*>(entry), entry_bytes)) {
            reason = slot.heap ? "a heap record is outside the list's view heap" : "a table descriptor is not bound";
            break;
        }
        std::memcpy(words, reinterpret_cast<const void*>(entry), entry_bytes);
        if (slot.texture) {
            // The title's descriptor memory carries the id of the view it created there. Nothing else is bound:
            // a null view would read zeros or drop the program's stores without a trace.
            ViewRecord record{};
            bool known = false, created = false;
            if (words[0] == view_token_magic) {
                AcquireSRWLockShared(&note_lock);
                known = words[1] < view_records().size();
                if (known) {
                    record = view_records()[words[1]];
                    created = view_materialized()[words[1]] != 0;
                }
                ReleaseSRWLockShared(&note_lock);
            }
            Wrapper* const wrapper = known ? as_wrapper(reinterpret_cast<const void*>(record.resource)) : nullptr;
            ID3D12Resource* const texture = !wrapper || wrapper->own().kind != resource_kind ? nullptr
                : output ? static_cast<ID3D12Resource*>(wrapper->own().real) : sampled_address_texture(wrapper);
            const D3D12_RESOURCE_DESC resource = texture ? texture->GetDesc() : D3D12_RESOURCE_DESC{};
            if (!output) {
                D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
                const bool described = record.desc_bytes >= sizeof(desc);
                if (described) std::memcpy(&desc, record.desc, sizeof(desc));
                if (desc.Format) desc.Format = desktop_format(desc.Format);
                words[3] = static_cast<std::uint32_t>(desc.Format ? desc.Format : resource.Format) | static_cast<std::uint32_t>(desc.ViewDimension) << 16;
                if (!known || record.slot != 18) reason = "an input texture entry is not the token of a shader resource view";
                else if (!texture || (resource.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE))
                    reason = "an input texture has no desktop texture to sample";
                else if (described ? desc.ViewDimension != D3D12_SRV_DIMENSION_TEXTURE2D
                                   : resource.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || resource.DepthOrArraySize != 1)
                    reason = "an input texture view is not 2D";
                // shortcut: a packed format needs the stand-in rules of the draw path; add them with its first user.
                else if (x2w_startup::packed_target_format(desc.Format)) reason = "an input texture view has a packed format";
                views[i] = {texture, 0, 0, true, true, described};
                std::memcpy(views[i].desc, &desc, sizeof(desc));
                views[i].wrapper = wrapper;
                continue;
            }
            D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
            const bool described = record.desc_bytes >= sizeof(desc);
            if (described) std::memcpy(&desc, record.desc, sizeof(desc));
            if (desc.Format) desc.Format = desktop_format(desc.Format);
            const DXGI_FORMAT format = desc.Format ? desc.Format : resource.Format;
            words[3] = static_cast<std::uint32_t>(format) | static_cast<std::uint32_t>(desc.ViewDimension) << 16;
            if (!known || record.slot != 19) reason = "an output texture entry is not the token of an unordered access view";
            else if (!created || !texture || !(resource.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
                reason = "an output texture view was not created on the desktop";
            else if (described ? desc.ViewDimension != D3D12_UAV_DIMENSION_TEXTURE2D || desc.Texture2D.PlaneSlice
                               : resource.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || resource.DepthOrArraySize != 1)
                reason = "an output texture view is not 2D";
            else if (x2w_startup::packed_target_format(format)) reason = "an output texture view has a packed format";
            else if (!float_output_format(format)) reason = "an output texture format is not a float or normalized one";
            // One state per resource here: a texture read and written by the same call is refused. Two outputs
            // of one texture (two mips) share the unordered access state.
            for (std::size_t input = 0; input < inputs && !reason; ++input)
                if (views[input].wrapper == wrapper) reason = "an output texture is also an input of the dispatch";
            views[i] = {texture, 0, 0, true, true, described};
            std::memcpy(views[i].desc, &desc, sizeof(desc));
            views[i].wrapper = wrapper;
            continue;
        }
        if (!output) {
            const IndirectBufferView view = indirect_table_buffer(self, words);
            if (!view.resource || view.offset % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT) reason = "an input is not a complete GPU-backed buffer range";
            views[i] = {view.resource, view.offset, view.bytes, !view.upload};
            continue;
        }
        // An output is written by the GPU: only a placed buffer that aliases the title's memory shows the result
        // to the title. shortcut: view tokens and copied placements are refused, add them with their first user.
        const u64 address = words[0] | (u64{words[1] & 0xffff} << 32);
        const UINT stride = (words[1] >> 16) & 0x3fff;
        const u64 bytes = u64{stride ? stride : 1} * words[2];
        const auto placed = words[0] == view_token_magic || !bytes || bytes % 4 || bytes > (16u << 20)
            ? PlacedVertexView{} : placed_buffer_range(self->own().list_device, address, bytes);
        const ResourceInfo* info = placed.alias ? resource_info(placed.owner) : nullptr;
        const u64 offset = placed.alias ? address - reinterpret_cast<u64>(placed.owner->own().title_memory) : 0;
        if (!info || info->initial_state != D3D12_RESOURCE_STATE_COMMON || offset % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT ||
            !(placed.alias->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
            reason = "an output is not an aliased placed buffer range with unordered access";
        for (std::size_t input = 0; input < inputs && !reason; ++input)
            if (views[input].resource == placed.alias) reason = "an output buffer is also an input";
        views[i] = {placed.alias, offset, bytes, true};
    }
    // A texture the title filled in memory is brought up to date first. An upload recorded before a later one
    // fails only syncs content, as a draw sampling that texture would.
    for (std::size_t i = 0; i < inputs && !reason; ++i)
        if (views[i].texture && !upload_texture(self, views[i].wrapper, views[i].resource)) {
            reason = "an input texture could not be uploaded";
            std::memset(words, 0, sizeof(words));
        }
    if (reason) {
        if (!tr->first_refusal) {
            tr->first_refusal = reason;
            std::memcpy(tr->refusal_words, words, sizeof(tr->refusal_words));
        }
        ReleaseSRWLockExclusive(&draw_lock);
        InterlockedIncrement(counter);
        // Dispatches that arrive before the offline translation must not use up the lines for bind refusals.
        static LONG untranslated, refusals;
        if (first_calls(counter == &compute_dispatches_refused ? refusals : untranslated, 16)) log("x2w-draw compute %016llx Dispatch(%u, %u, %u) refused: %s", key, x, y, z, reason);
        return true;
    }
    const UINT n = static_cast<UINT>(count);
    UINT first = static_cast<UINT>(InterlockedExchangeAdd(&draw_heap_cursor, static_cast<LONG>(n))) % draw_heap_size;
    if (first + n > draw_heap_size) {
        first = 0;
        InterlockedExchange(&draw_heap_cursor, static_cast<LONG>(n));
    }
    D3D12_CPU_DESCRIPTOR_HANDLE dest{draw_heap->GetCPUDescriptorHandleForHeapStart().ptr + std::size_t{first} * draw_heap_increment};
    for (std::size_t i = 0; i < count; ++i, dest.ptr += draw_heap_increment) {
        const View& view = views[i];
        if (i < inputs && view.texture) {
            // A compute list accepts no pixel-shader or depth state in a transition.
            require_state(self, view.resource, type == D3D12_COMMAND_LIST_TYPE_COMPUTE ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                : sampled_state(self, view.resource) | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            device_of(the_device)->CreateShaderResourceView(view.resource,
                view.described ? reinterpret_cast<const D3D12_SHADER_RESOURCE_VIEW_DESC*>(view.desc) : nullptr, dest);
            continue;
        }
        if (i < inputs) {
            if (view.transition) require_state(self, view.resource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            buffer_view(dest, view.resource, view.offset, static_cast<std::size_t>(view.bytes));
            continue;
        }
        require_state(self, view.resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (view.texture) {
            device_of(the_device)->CreateUnorderedAccessView(view.resource, nullptr,
                view.described ? reinterpret_cast<const D3D12_UNORDERED_ACCESS_VIEW_DESC*>(view.desc) : nullptr, dest);
            continue;
        }
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
        desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        desc.Buffer.FirstElement = view.offset / 4;
        desc.Buffer.NumElements = static_cast<UINT>(view.bytes / 4);
        desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device_of(the_device)->CreateUnorderedAccessView(view.resource, nullptr, &desc, dest);
    }
    std::uint32_t values[32] = {};
    for (std::size_t i = 0; i < tr->uds.size(); ++i) values[i] = pass.compute_constants.word(tr->uds[i].a, tr->uds[i].b);
    if (!pass.heap_set) {
        x2w_startup::bind_translated_heaps(list, draw_heap, nullptr);
        pass.heap_set = true;
        pass.bound_sampler_heap = nullptr;
    }
    list->SetPipelineState(tr->pipeline.Get());
    list->SetComputeRootSignature(tr->root.Get());
    if (!tr->uds.empty()) list->SetComputeRoot32BitConstants(0, static_cast<UINT>(tr->uds.size()), values, 0);
    if (n) list->SetComputeRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE{draw_heap->GetGPUDescriptorHandleForHeapStart().ptr + u64{first} * draw_heap_increment});
    list->Dispatch(x, y, z);
    for (std::size_t i = inputs; i < count; ++i) {  // a later dispatch on this list reads what this one wrote
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = views[i].resource;
        list->ResourceBarrier(1, &barrier);
    }
    // The translated root replaced the desktop compute root of the title and its arguments.
    if (pass.desktop_compute_root) {
        list->SetComputeRootSignature(pass.desktop_compute_root.Get());
        for (UINT parameter = 0; parameter < pass.desktop_compute_upload_srvs.size(); ++parameter)
            if (const auto address = pass.desktop_compute_upload_srvs[parameter]) list->SetComputeRootShaderResourceView(parameter, address);
    }
    ++tr->translated;
    ReleaseSRWLockExclusive(&draw_lock);
    static LONG checks;  // as translated draws do: the queue's execute drains the rest
    if ((InterlockedIncrement(&checks) & 63) == 1) check_device("translated dispatch", 14);
    InterlockedIncrement(&compute_dispatches_translated);
    static LONG logged;
    if (first_calls(logged, 8))
        log("x2w-draw compute %016llx Dispatch(%u, %u, %u) translated: entry [%x %x %x %x], %zu inputs, %zu outputs", key, x, y, z,
            values[0], values[1], values[2], values[3], inputs, count - inputs);
    return true;
}

// With the periodic totals: refused dispatches by the first refusal reason of their key, then every key by
// dispatch count. The per-call log lines above share one small budget, so they cannot rank keys.
void log_compute_census()
{
    struct Reason { const char* text; u64 keys, dispatches; };
    std::vector<Reason> reasons;
    std::vector<ComputeTranslation> keys;
    AcquireSRWLockShared(&draw_lock);
    for (const auto& item : compute_translations()) {
        const ComputeTranslation& tr = item.second;
        ComputeTranslation copy;
        copy.key = tr.key;
        copy.dispatches = tr.dispatches;
        copy.translated = tr.translated;
        copy.first_refusal = tr.first_refusal;
        std::memcpy(copy.refusal_words, tr.refusal_words, sizeof(copy.refusal_words));
        keys.push_back(copy);
        if (!tr.first_refusal) continue;
        auto found = std::find_if(reasons.begin(), reasons.end(), [&](const Reason& r) { return !std::strcmp(r.text, tr.first_refusal); });
        if (found == reasons.end()) found = reasons.insert(reasons.end(), Reason{tr.first_refusal, 0, 0});
        ++found->keys;
        found->dispatches += tr.dispatches - tr.translated;
    }
    ReleaseSRWLockShared(&draw_lock);
    if (keys.empty()) return;
    std::sort(reasons.begin(), reasons.end(), [](const Reason& a, const Reason& b) { return a.dispatches > b.dispatches; });
    std::sort(keys.begin(), keys.end(), [](const ComputeTranslation& a, const ComputeTranslation& b) { return a.dispatches > b.dispatches; });
    log("x2w-render compute census: %zu keys dispatched", keys.size());
    for (const Reason& r : reasons)
        log("x2w-render compute census reason \"%s\": %llu keys, %llu dispatches", r.text, r.keys, r.dispatches);
    for (std::size_t i = 0; i < keys.size(); ++i)
        log("x2w-render compute census key %016llx: %llu dispatches, %llu translated, first refusal: %s [%08x %08x %08x %08x]",
            keys[i].key, keys[i].dispatches, keys[i].translated, keys[i].first_refusal ? keys[i].first_refusal : "none",
            keys[i].refusal_words[0], keys[i].refusal_words[1], keys[i].refusal_words[2], keys[i].refusal_words[3]);
}

// Fixtures: the census without waiting for the frame that logs it.
extern "C" __declspec(dllexport) void x2w_d3d12_compute_census() { log_compute_census(); }
