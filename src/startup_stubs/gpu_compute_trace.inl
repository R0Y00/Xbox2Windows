// Only the selected shader in a numbered trace request can record copies.
using ComputeCapture = x2w_compute_trace::Capture;
std::mutex compute_capture_lock;
std::array<std::weak_ptr<ComputeCapture>, 2> compute_capture_slots;
std::atomic<LONG> compute_capture_pending{0};
x2w_gpu::CaptureSelection capture_selection;
x2w_gpu::CapturePoint requested_point;
void arm_compute_capture(LONG id, const TraceStageKey& key, const TraceStageKey& second, x2w_gpu::CapturePoint point)
{
    std::lock_guard<std::mutex> guard(compute_capture_lock);
    requested_point = point;
    capture_selection = {key ? id : 0, {key ? x2w_gpu::stage_key_number(*key) : std::nullopt,
        second ? x2w_gpu::stage_key_number(*second) : std::nullopt}, bool(second)};
    compute_capture_pending.store(key ? id : 0);
}
LONG match_compute_capture(const char* key)
{
    if (!key || !compute_capture_pending.load()) return 0;
    std::lock_guard<std::mutex> guard(compute_capture_lock);
    return capture_selection.matches(capture_selection.id, key) ? capture_selection.id : 0;
}

// Readback buffers remain owned by their recording across replays. A second reference
// survives Reset until every queue that actually submitted that recording has completed.
std::vector<std::shared_ptr<ComputeCapture>> compute_capture_flights; // accessed under draw_lock
std::atomic<bool> compute_capture_in_flight{false};

void compute_capture_copy(Wrapper* self, ID3D12GraphicsCommandList* list, ComputeCapture& cap, bool post)
{
    for (auto& image : cap.images) {
        if (post && !image.binding.write) continue;
        auto* source = image.source.Get();
        auto& states = pass_of(self).states;
        const auto* it = states.find(source);
        // Every enrolled image was checked before any copy on this dispatch.
        if (!it) continue;
        if (image.full_depth_plane) {
            const auto prior = *it;
            require_planes(self, source, image.binding.plane ? prior.depth : D3D12_RESOURCE_STATE_COPY_SOURCE,
                image.binding.plane ? D3D12_RESOURCE_STATE_COPY_SOURCE : prior.stencil);
            image.record(list, post);
            require_planes(self, source, prior.depth, prior.stencil);
            continue;
        }
        const auto previous = it->depth;
        require_state(self, source, D3D12_RESOURCE_STATE_COPY_SOURCE);
        image.record(list, post);
        require_state(self, source, previous);
    }
}
x2w_compute_trace::Image::Focus capture_focus()
{
    static const auto focus = []() {
        x2w_compute_trace::Image::Focus point{500,500}, parsed{};
        char value[32]{}, extra{};
        const auto n = GetEnvironmentVariableA("X2W_TRACE_PIXEL_CENTER", value, sizeof(value));
        if (n && n < sizeof(value) && sscanf_s(value, "%u,%u%c", &parsed.x, &parsed.y, &extra, 1u) == 2 &&
            parsed.x <= 1000 && parsed.y <= 1000) point = parsed;
        return point;
    }();
    if (requested_point) return {(*requested_point)[0], (*requested_point)[1]};
    return focus;
}
void compute_capture_before(Wrapper* self, ID3D12GraphicsCommandList* list)
{
    if (!compute_capture_on() || !compute_capture_pending.load()) return;
    auto& s = state_of(self);
    if (!s.trace_request || !s.trace_bindings.writes) return;
    std::lock_guard<std::mutex> guard(compute_capture_lock);
    if (!s.trace_shader || !capture_selection.matches(s.trace_request, s.trace_shader->key)) return;
    auto slot = std::find_if(compute_capture_slots.begin(), compute_capture_slots.end(), [](auto& weak) { return weak.expired(); });
    if (slot == compute_capture_slots.end()) return;
    auto cap = std::make_shared<ComputeCapture>();
    cap->request = s.trace_request;
    cap->frame = InterlockedCompareExchange(&frame_number, 0, 0) + 1;
    cap->key = s.trace_shader->key;
    cap->paired = capture_selection.paired;
    for (UINT i = 0; i < s.trace_bindings.count; ++i) {
        const auto& b = s.trace_bindings.values[i];
        const auto& states = pass_of(self).states;
        const auto* found = states.find(b.texture);
        if (!found) continue;
        const auto state = b.plane ? found->stencil : found->depth;
        if (b.write ? state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS :
                      !(state & D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)) continue;
        x2w_compute_trace::Image image;
        const auto& focus = capture_focus();
        if (image.init(device_of(the_device), b, &focus)) cap->images.push_back(std::move(image));
    }
    if (std::none_of(cap->images.begin(), cap->images.end(), [](const auto& i) { return i.binding.write; })) return;
    capture_selection.complete(cap->key);
    compute_capture_pending.store(capture_selection.id);
    *slot = cap;
    s.trace_current = cap;
    s.trace_captures.push_back(cap);
    compute_capture_copy(self, list, *cap, false);
    log("x2w-compute-capture request %ld frame %ld shader=%s list=%p sampled %zu selected textures",
        cap->request, cap->frame, cap->key.c_str(), static_cast<void*>(self), cap->images.size());
}
void compute_capture_after(Wrapper* self, ID3D12GraphicsCommandList* list)
{
    if (!compute_capture_on()) return;
    auto& s = state_of(self);
    if (!s.trace_current) return;
    compute_capture_copy(self, list, *s.trace_current, true);
    s.trace_current.reset();
}

void pixel_capture_before(Wrapper* self, ID3D12GraphicsCommandList* list, const char* command)
{
    auto& s = state_of(self);
    if (!s.pixel_capture_shader || list->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT) return;
    auto& pass = pass_of(self);
    if (pass.target_count != 1 || !pass.targets[0]) return;
    const auto dest = pass.targets[0]->GetDesc();
    if (dest.MipLevels != 1 || dest.DepthOrArraySize != 1 || dest.SampleDesc.Count != 1) return;
    std::lock_guard<std::mutex> guard(compute_capture_lock);
    if (!capture_selection.matches(s.trace_request, s.pixel_capture_shader->key)) return;
    auto seat = std::find_if(compute_capture_slots.begin(), compute_capture_slots.end(), [](auto& w){ return w.expired(); });
    if (seat == compute_capture_slots.end()) return;
    auto cap = std::make_shared<ComputeCapture>();
    const auto stage = [&](x2w_compute_trace::Binding b) {

        const auto& focus = capture_focus();
        const auto d = b.texture->GetDesc();
        const auto f = b.view_format == DXGI_FORMAT_UNKNOWN ? d.Format : b.view_format;
        const bool ds = (d.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0;
        if (d.SampleDesc.Count != 1 || d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER ||
            (ds ? (b.write || b.plane > 1 || !x2w_compute_trace::depth_format(d.Format)) : !x2w_compute_trace::supported(f))) return false;
        const auto* state = pass.states.find(b.texture);
        if (!state || !((b.plane ? state->stencil : state->depth) &
            (b.write ? D3D12_RESOURCE_STATE_RENDER_TARGET : D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE))) return false;
        x2w_compute_trace::Image image;
        if (!image.init(device_of(the_device), b, &focus)) return false;
        cap->images.push_back(std::move(image));
        return true;
    };
    if (!stage({pass.targets[0], 0, 0, true, false, 0, 0, pass.target_formats[0]})) return;
    for (const auto& b : s.pixel_bindings) stage(b);
    cap->pixel = true;
    cap->request = s.trace_request;
    cap->paired = capture_selection.paired;
    cap->frame = InterlockedCompareExchange(&frame_number, 0, 0) + 1;
    cap->key = s.pixel_capture_shader->key;
    capture_selection.complete(cap->key);
    compute_capture_pending.store(capture_selection.id);
    *seat = cap;
    s.trace_current = cap;
    s.trace_captures.push_back(cap);
    compute_capture_copy(self, list, *cap, false);
    const auto* p = s.pixel_capture_pipeline;
    log("x2w-pixel-capture request=%ld frame=%ld shader=%s list=%p command=%s rtv=%p color_mask=%x depth_test=%d depth_write=%d scene inputs captured=%zu/%zu",
        cap->request, cap->frame, cap->key.c_str(), static_cast<void*>(self), command, pass.targets[0], p->color_mask,
        int(p->depth_test), int(p->depth_write), cap->images.size() - 1, s.pixel_bindings.size());
}
void pixel_capture_after(Wrapper* self, ID3D12GraphicsCommandList* list)
{
    auto& cap = state_of(self).trace_current;
    if (!cap || !cap->pixel) return;
    compute_capture_copy(self, list, *cap, true);
    cap.reset();
}

struct MaterialFeedback {
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap;
    x2w_material_trace::Words words{};
    x2w_material::Mask fields{};
};
struct MaterialSample {
    ID3D12DescriptorHeap* heap{};
    ID3D12Resource* source{};
    UINT64 offset{};
    LONG frame{};
    x2w_material::Mask fields{};
    std::weak_ptr<ComputeCapture> capture;
};
std::mutex material_feedback_lock;
std::deque<MaterialFeedback> material_ready;
std::array<MaterialSample, 192> material_samples{};
void collect_material_feedback(ComputeCapture& cap)
{
    x2w_material_trace::Words words{};
    D3D12_RANGE range{0, x2w_material::byte_count(cap.material.fields)}, none{};
    void* ptr{};
    if (FAILED(cap.material.readback->Map(0, &range, &ptr))) return;
    std::memcpy(words.data(), ptr, range.End);
    cap.material.readback->Unmap(0, &none);
    std::lock_guard<std::mutex> guard(material_feedback_lock);
    if (material_ready.size() == material_samples.size()) material_ready.pop_front();
    material_ready.push_back({cap.feedback_heap, words, cap.material.fields});
}
void prepare_material(Wrapper* self, ID3D12GraphicsCommandList* list, GpuListState& s)
{
    static const bool lazy = x2w_local::env_on("X2W_GPU_LAZY_DETILE");
    if (!lazy || !s.host_heaps[0]) return;
    std::array<MaterialFeedback, 3> ready;
    unsigned n = 0;
    {
        std::lock_guard<std::mutex> guard(material_feedback_lock);
        for (auto it = material_ready.begin(); it != material_ready.end() && n < ready.size();) {
            if (it->heap.Get() == s.host_heaps[0]) {
                ready[n++] = std::move(*it);
                it = material_ready.erase(it);
            } else ++it;
        }
    }
    unsigned made = 0;
    for (unsigned i = 0; i < n; ++i) made += gpu_materialize(self, s.material_heap, ready[i].words, ready[i].fields);
    const auto fields = s.material_shader->material_fields;
    const auto r = gpu_material_record(s.material_heap, s.material_raw_slot, s.material_base, fields);
    if (!r.source) return;
    x2w_material_trace::Words words;
    const bool cpu = x2w_material_trace::cpu_words(r, words);
    if (cpu) made += gpu_materialize(self, s.material_heap, words, fields);
    if (made) {
        static LONG count;
        if (first_calls(count, 96)) log("x2w-material resident frame=%ld record=%u fenced_records=%u upload_heap=%d slots=%u",
            frame_number, s.material_base, n, int(cpu), made);
    }
    if (cpu) return;
    const auto* state = pass_of(self).states.find(r.source.Get());
    if (!state) return;
    const auto before = state->depth;
    const LONG frame = InterlockedCompareExchange(&frame_number, 0, 0);
    std::shared_ptr<ComputeCapture> cap;
    {
        std::lock_guard<std::mutex> guard(material_feedback_lock);
        MaterialSample* dest = nullptr;
        unsigned recording = 0;
        for (auto& entry : material_samples) {
            const bool active = !entry.capture.expired();
            recording += entry.frame == frame && entry.source;
            if (entry.heap == s.host_heaps[0] && entry.source == r.source.Get() && entry.offset == r.offset &&
                (entry.fields & fields) == fields &&
                (active || frame - entry.frame < 30)) return;
            if (!active && (!dest || entry.frame < dest->frame)) dest = &entry;
        }
        if (!dest || recording >= 12) return;
        cap = std::make_shared<ComputeCapture>();
        *dest = {s.host_heaps[0], r.source.Get(), r.offset, frame, fields, cap};
    }
    D3D12_HEAP_PROPERTIES props{D3D12_HEAP_TYPE_READBACK};
    props.CreationNodeMask = props.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = x2w_material::byte_count(fields);
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(device_of(the_device)->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&cap->material.readback)))) return;
    cap->feedback_heap = s.host_heaps[0];
    cap->feedback_source = r.source;
    cap->material.fields = fields;
    s.trace_captures.push_back(cap);
    x2w_material_trace::copy_record(list, r, cap->material.readback.Get(), before,
        [&](D3D12_RESOURCE_STATES next) { require_state(self, r.source.Get(), next); });
}

void material_capture(Wrapper* self, ID3D12GraphicsCommandList* list, GpuListState& s, ID3D12PipelineState* graphics)
{
    static std::mutex capture_lock;
    static std::atomic<LONG> previous{};
    const LONG request = gpu_trace_request_id;
    if (!request || request == previous.load() || !s.host_heaps[0]) return;
    std::lock_guard<std::mutex> guard(capture_lock);
    if (request == previous || request != gpu_trace_request_id) return;
    const UINT raw = s.material_raw_slot | (x2w_material_trace::null_index + 1);
    if (raw >= s.host_heaps[0]->GetDesc().NumDescriptors) return;
    auto* resource = gpu_heap_kept_at(s.material_heap, s.material_raw_slot);
    if (resource) {
        const auto& states = pass_of(self).states;
        const auto* found = states.find(resource);
        if (!found || !(found->depth & D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)) return;
    }
    previous.store(request);
    auto cap = std::make_shared<ComputeCapture>();
    auto& p = cap->material;
    if (!x2w_material_trace::init(device_of(the_device), p)) {
        log("x2w-material request=%ld probe initialization failed", request);
        return;
    }
    cap->request = request;
    cap->frame = gpu_trace_frame;
    cap->key = s.material_shader->key;
    p.fields = s.material_shader->material_fields;
    p.raw_slot = s.material_raw_slot;
    p.base = s.material_base;
    gpu_material_slots(s.material_heap, p.slots);
    s.trace_captures.push_back(cap);
    x2w_material_trace::record(list, p, gpu_heap_gpu_handle(s.material_heap, raw));
    list->SetPipelineState(graphics);
    s.host_roots[1] = nullptr;
    log("x2w-material request=%ld frame=%ld list=%p captured material table=%u record=%u", request,
        cap->frame, static_cast<void*>(self), p.raw_slot, p.base);
}

void compute_capture_save(ComputeCapture& cap)
{
    if (cap.feedback_heap) {
        collect_material_feedback(cap);
        return;
    }
    if (cap.material.readback) {
        auto& p = cap.material;
        D3D12_RANGE range{0, x2w_material::byte_count(p.fields)}, none{};
        void* data{};
        if (FAILED(p.readback->Map(0, &range, &data))) return;
        for (UINT field = 0; field < range.End; field += 4) {
            if (!(p.fields & (1u << (field / 4)))) continue;
            const UINT word = static_cast<const UINT*>(data)[field / 4];
            const UINT index = x2w_material_trace::index(word);
            const auto it = std::find_if(p.slots.begin(), p.slots.end(), [index](auto& slot) { return slot.index == index; });
            const x2w_material_trace::Slot empty{};
            const auto& slot = it == p.slots.end() ? empty : *it;
            log("x2w-material request=%ld frame=%ld shader=%s table=%u record=%u field=%u word=%08x index=%u "
                "observed_at_record=%d host=%llx dimension=%u size=%ux%u format=%u view_format=%u address=%llx "
                "lazy_pending=%d late_pending=%d feedback_retry=%s words=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x",
                cap.request, cap.frame, cap.key.c_str(), p.raw_slot, p.base, field, word, index, it != p.slots.end(),
                slot.host, slot.dimension, slot.width, slot.height, slot.format, slot.view_format, slot.texture_address,
                int(slot.lazy), int(slot.late), slot.material_retry, slot.words[0], slot.words[1], slot.words[2], slot.words[3],
                slot.words[4], slot.words[5], slot.words[6], slot.words[7]);
        }
        p.readback->Unmap(0, &none);
        std::vector<x2w_material_trace::Slot>{}.swap(p.slots);
        return;
    }
    wchar_t root[MAX_PATH]{};
    const DWORD len = GetEnvironmentVariableW(L"X2W_TRACE_TARGET_DIR", root, MAX_PATH);
    if (!len || len >= MAX_PATH) return;
    auto dir = std::filesystem::path(root) / ((cap.pixel ? L"pixel-request-" : L"compute-request-") + std::to_wstring(cap.request));
    if (cap.paired) dir += "-" + cap.key;
    std::error_code ec;
    if (!std::filesystem::create_directory(dir, ec)) {
        log("x2w-compute-capture request %ld needs a fresh directory, error %d", cap.request, ec.value());
        return;
    }
    x2w_compute_trace::IndirectArguments::Snapshot args{};
    if (cap.indirect.read(args)) {
        const auto& indirect = cap.indirect;
        const UINT active = (std::min)(indirect.copied, args.count);
        FILE* table = nullptr;
        _wfopen_s(&table, (dir / L"indirect-arguments.tsv").c_str(), L"wbx");
        if (table) {
            std::fprintf(table, "kind\tstride\tmaximum\tcount_buffer\teffective\tsampled\n%u\t%u\t%u\t%u\t%u\t%u\n",
                indirect.kind, indirect.stride, indirect.maximum, unsigned(indirect.counted), args.count, active);
            if (indirect.kind == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW)
                std::fputs("record\tvertex_count\tinstance_count\tstart_vertex\tstart_instance\n", table);
            else if (indirect.kind == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED)
                std::fputs("record\tindex_count\tinstance_count\tstart_index\tbase_vertex\tstart_instance\n", table);
            else std::fputs("record\tgroups_x\tgroups_y\tgroups_z\n", table);
        }
        log("x2w-pixel-capture request=%ld frame=%ld shader=%s indirect kind=%u stride=%u maximum=%u counted=%u effective=%u sampled=%u",
            cap.request, cap.frame, cap.key.c_str(), indirect.kind, indirect.stride, indirect.maximum, unsigned(indirect.counted), args.count, active);
        for (UINT i = 0; i < active; ++i) {
            const auto& r = args.records[i];
            if (table) {
                std::fprintf(table, "%u", i);
                for (UINT field = 0; field < indirect.fields; ++field) {
                    if (indirect.kind == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED && field == 3)
                        std::fprintf(table, "\t%d", static_cast<INT>(r[field]));
                    else std::fprintf(table, "\t%u", r[field]);
                }
                std::fputc('\n', table);
            }
            log("x2w-pixel-capture request=%ld frame=%ld shader=%s indirect record=%u fields=%u words=%u,%u,%u,%u,%u",
                cap.request, cap.frame, cap.key.c_str(), i, indirect.fields, r[0], r[1], r[2], r[3], r[4]);
        }
        if (table) std::fclose(table);
    }
    for (const auto& image : cap.images) {
        const auto& b = image.binding;
        for (unsigned phase = 0; phase < (b.write ? 2u : 1u); ++phase) {
            auto* buffer = phase ? image.post.Get() : image.pre.Get();
            const auto& area = image.footprint.Footprint;
            const D3D12_RANGE range{0, static_cast<SIZE_T>(image.size)};
            void* mapped = nullptr;
            if (FAILED(buffer->Map(0, &range, &mapped))) continue;
            const auto* pixels = static_cast<const unsigned char*>(mapped) + image.footprint.Offset;
            wchar_t label[96]{};
            const wchar_t* role = b.write ? (cap.pixel ? L"rtv" : L"uav") : L"srv";
            swprintf_s(label, L"%ls-%u-%u-%ls", role, b.slot, b.offset, phase ? L"post" : L"pre");
            const auto stem = dir / label;
            if (cap.pixel) log("x2w-pixel-capture request=%ld shader=%s role=%ls slot=%u offset_dwords=%u "
                "mip=%u first_slice=%u plane=%u resource_format=%u view_format=%u", cap.request, cap.key.c_str(), role, b.slot, b.offset,
                b.mip, b.slice, b.plane, UINT(image.source->GetDesc().Format), UINT(b.view_format));
            const UINT total_rows = image.height() * area.Depth;
            const UINT64 row_bytes = image.output_row_bytes();
            UINT64 hash = 14695981039346656037ull, nonzero = 0;
            for (UINT y = 0; y < total_rows; ++y) {
                const auto* row = image.row(pixels, y % image.height(), y / image.height());
                for (UINT64 x = 0; x < row_bytes; ++x) {
                    hash = (hash ^ row[x]) * 1099511628211ull;
                    nonzero += row[x] != 0;
                }
            }
            FILE* file = nullptr;
            bool raw = false, bmp = false;
            if (_wfopen_s(&file, (stem.wstring() + L".bin").c_str(), L"wbx") == 0) {
                raw = true;
                for (UINT y = 0; y < total_rows && raw; ++y)
                    raw = std::fwrite(image.row(pixels, y % image.height(), y / image.height()), 1, SIZE_T(row_bytes), file) == row_bytes;
                raw = std::fclose(file) == 0 && raw;
            }
            if (x2w_draw_bmp::trace_supported(area.Format)) {
                bmp = true;
                for (UINT z = 0; z < area.Depth; ++z) {
                    if (z != 0 && z != area.Depth / 2 && z != area.Depth - 1) continue;
                    const auto suffix = image.source_depth > 1 ? L"-z" + std::to_wstring(image.source_z(z)) : L"";
                    if (_wfopen_s(&file, (stem.wstring() + suffix + L".bmp").c_str(), L"wbx") != 0) { bmp = false; continue; }
                    const auto* plane = image.row(pixels, 0, z);
                    const bool ok = x2w_draw_bmp::write(file, plane, image.width(), image.height(), area.RowPitch, area.Format);
                    bmp = (std::fclose(file) == 0 && ok) && bmp;
                }
            }
            const D3D12_RANGE none{0,0};
            buffer->Unmap(0, &none);
            log("x2w-%s-capture request=%ld frame=%ld shader=%s slot=%u offset_dwords=%u %ls %s resource=%p "
                "fmt=%u box=%u,%u %ux%u depth=%u/%u layout=%s rowbytes=%llu fnv64=%016llx nonzero_bytes=%llu raw=%u bmp=%u",
                cap.pixel ? "pixel" : "compute", cap.request, cap.frame, cap.key.c_str(),
                b.slot, b.offset, role, phase ? "post" : "pre", static_cast<void*>(b.texture),
                UINT(area.Format), image.box.left, image.box.top, image.width(), image.height(), area.Depth, image.source_depth,
                image.full_depth_plane ? (b.plane ? "stencil-plane-1" : "depth-plane-0") :
                image.sparse ? "first-middle-last" : "contiguous", row_bytes, hash, nonzero, raw, bmp);
        }
    }
}

void compute_capture_submitted(const std::shared_ptr<ComputeCapture>& cap, ID3D12Fence* fence, UINT64 value)
{
    auto& flights = compute_capture_flights;
    if (std::find(flights.begin(), flights.end(), cap) == flights.end()) flights.push_back(cap);
    compute_capture_in_flight.store(true);
    if (!fence) {
        cap->unfenced = true;
        log("x2w-compute-capture request %ld queue fence failed; retaining buffers", cap->request);
        return;
    }
    for (auto& f : cap->fences)
        if (f.first == fence) { f.second = value; return; }
    cap->fences.emplace_back(fence, value);  // queue_fences owns the fence for the lifetime of the device
}
// Called only under draw_lock, which serializes every native submission and therefore replay.
void compute_capture_poll_locked()
{
    auto& flights = compute_capture_flights;
    for (auto it = flights.begin(); it != flights.end();) {
        auto& cap = **it;
        if (cap.unfenced) { ++it; continue; }
        bool ready = true, removed = false;
        for (const auto& f : cap.fences) {
            const auto completed = f.first->GetCompletedValue();
            if (completed == UINT64_MAX) removed = true;
            else if (completed < f.second) ready = false;
        }
        if (!ready) { ++it; continue; }
        if (!removed && !cap.published) compute_capture_save(cap);
        cap.published = true;
        cap.fences.clear();
        it = flights.erase(it);
    }
    compute_capture_in_flight.store(!flights.empty());
}
