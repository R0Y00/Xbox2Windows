// Opt-in diagnostic for writes to the title-selected presentation texture.
extern LONG frame_number;

struct TraceUavSelection {
    u64 key = 0;
    int slot = -1;
    constexpr int match(u64 candidate) const { return key == candidate ? slot : -1; }
};

constexpr TraceUavSelection parse_trace_uav(std::string_view value)
{
    if (value.size() < 18 || value.size() > 19 || value[16] != ',') return {};
    u64 key = 0;
    for (char c : value.substr(0, 16)) {
        unsigned digit = 0;
        if (c >= '0' && c <= '9') digit = c - '0';
        else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
        else return {};
        key = (key << 4) | digit;
    }
    int slot = 0;
    for (char c : value.substr(17)) {
        if (c < '0' || c > '9') return {};
        slot = 10 * slot + c - '0';
    }
    return slot < 64 ? TraceUavSelection{key, slot} : TraceUavSelection{};
}

static_assert(parse_trace_uav("0123456789abCDef,63").match(0x0123456789abcdefULL) == 63);
static_assert(parse_trace_uav("0123456789abcdef,20").match(1) == -1);
static_assert(parse_trace_uav("0123456789abcdef,64").slot == -1);
static_assert(parse_trace_uav("0123456789abcdef,2x").slot == -1);

class RedProducers {
    using Resource = Microsoft::WRL::ComPtr<ID3D12Resource>;
    using Fence = Microsoft::WRL::ComPtr<ID3D12Fence>;
    static constexpr UINT64 after_offset = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
    static constexpr std::size_t history_limit = 64;
    struct Entry {
        ID3D12GraphicsCommandList* list;
        Resource readback, surface;
        Fence fence;
        ID3D12CommandQueue* queue = nullptr;
        LONG record_frame, submit_frame = 0;
        unsigned attachment, ordinal, submit_order = 0;
        const char* operation;
        u64 key;
        bool live = true;
        bool pending = false;
        bool ambiguous = false;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT64 bytes = 0, row_bytes = 0;
        UINT rows = 0;
    };
    struct History {
        const void* list;
        const void* surface;
        LONG frame;
        unsigned attachment;
        const char* operation;
        u64 key;
    };
    std::mutex lock_;
    std::vector<Resource> surfaces_;
    std::vector<Entry> entries_;
    LONG first_ = 0, last_ = 0;
    LONG auto_min_ = 0, auto_count_ = 0;
    volatile LONG auto_first_ = 0, auto_last_ = 0;
    unsigned target_rgb_[3]{255, 0, 0};
    bool auto_last_match_ = false;
    History history_[history_limit]{};
    std::size_t history_next_ = 0, history_count_ = 0;
    unsigned submit_order_ = 0;
    bool full_logged_ = false;
    TraceUavSelection uav_;
    bool pair_ = false;
    unsigned uav_samples_ = 0;
    LONG uav_last_frame_ = 0;

public:
    struct Disabled {};
    explicit RedProducers(Disabled) {}

    RedProducers()
    {
        char value[48]{};
        const DWORD size = GetEnvironmentVariableA("X2W_CX_TRACE_UAV", value, sizeof(value));
        if (size && size < sizeof(value)) uav_ = parse_trace_uav({value, size});
        pair_ = uav_.slot >= 0 && x2w_local::env_on("X2W_CX_TRACE_PAIR");
        if (GetEnvironmentVariableA("X2W_RED_PRODUCER_FRAMES", value, sizeof(value)) &&
            std::sscanf(value, "%ld,%ld", &first_, &last_) == 2 && first_ >= 1 && last_ >= first_) {
            log("x2w-red producer capture frames %ld..%ld; max 512 center texels from known present targets", first_, last_);
        } else {
            first_ = last_ = 0;
        }
        if (GetEnvironmentVariableA("X2W_RED_PRODUCER_AUTO", value, sizeof(value)) &&
            std::sscanf(value, "%ld,%ld", &auto_min_, &auto_count_) == 2 && auto_min_ >= 1 &&
            auto_count_ >= 1 && auto_count_ <= 120) {
            const DWORD rgb_bytes = GetEnvironmentVariableA("X2W_RED_PRODUCER_RGB", value, sizeof(value));
            if (rgb_bytes) {
                unsigned rgb[3]{};
                char trailing;
                if (rgb_bytes >= sizeof(value) || std::sscanf(value, "%3u,%3u,%3u%c", &rgb[0], &rgb[1], &rgb[2], &trailing) != 3 ||
                    rgb[0] > 255 || rgb[1] > 255 || rgb[2] > 255) {
                    auto_min_ = auto_count_ = 0;
                    log("x2w-red producer auto disabled: X2W_RED_PRODUCER_RGB needs three decimal components from 0 to 255");
                    return;
                }
                std::copy_n(rgb, 3, target_rgb_);
            }
            log("x2w-red producer waits for RGB %u,%u,%u at or after present %ld then captures %ld future frames",
                target_rgb_[0], target_rgb_[1], target_rgb_[2], auto_min_, auto_count_);
        } else {
            auto_min_ = auto_count_ = 0;
        }
    }

    bool enabled() const { return first_ || auto_count_ || uav_.slot >= 0; }
    bool matches(const unsigned char* rgba) const
    {
        return rgba[0] == target_rgb_[0] && rgba[1] == target_rgb_[1] && rgba[2] == target_rgb_[2];
    }
    bool interested() const
    {
        const LONG next = frame_number + 1;
        return (next >= first_ && next <= last_) || (next >= auto_first_ && next <= auto_last_);
    }
    bool may_record() const
    {
        return interested() || (auto_count_ && !auto_last_ && frame_number + 1 >= auto_min_);
    }

    void observe(LONG64 sampled_frame, LONG published_frame, ID3D12Resource* surface, bool matched)
    {
        if (!auto_count_ || auto_last_) return;
        std::lock_guard<std::mutex> guard(lock_);
        const bool became_matched = matched && !auto_last_match_;
        auto_last_match_ = matched;
        if (auto_last_ || !became_matched || sampled_frame < auto_min_ || published_frame > LONG_MAX - auto_count_) return;
        InterlockedExchange(&auto_first_, published_frame + 1);
        InterlockedExchange(&auto_last_, published_frame + auto_count_);
        log("x2w-red auto armed sampled_present=%lld published_at=%ld target=%p capture=%ld..%ld rgb=%u,%u,%u",
            sampled_frame, published_frame, surface, static_cast<LONG>(auto_first_), static_cast<LONG>(auto_last_),
            target_rgb_[0], target_rgb_[1], target_rgb_[2]);
        for (std::size_t i = 0; i < history_count_; ++i) {
            const auto& e = history_[(history_next_ + history_limit - history_count_ + i) % history_limit];
            log("x2w-red history record=%ld list=%p target=%p op=%s key=%016llx rt=%u",
                e.frame, e.list, e.surface, e.operation, e.key, e.attachment);
        }
    }

    void present(ID3D12Resource* surface)
    {
        if ((!first_ && !auto_count_) || !surface) return;
        std::lock_guard<std::mutex> guard(lock_);
        const auto found = std::find_if(surfaces_.begin(), surfaces_.end(), [=](const Resource& r) { return r.Get() == surface; });
        if (found == surfaces_.end() && surfaces_.size() < 8) {
            surfaces_.emplace_back(surface);
            const auto desc = surface->GetDesc();
            log("x2w-red present target=%p size=%llux%u format=%d", surface, desc.Width, desc.Height, desc.Format);
        }
    }

    unsigned before(Wrapper* owner, ID3D12GraphicsCommandList* list, ID3D12Resource* surface,
                    const char* operation, u64 key, unsigned attachment)
    {
        if (!may_record() || !surface || list->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT) return 0;
        std::lock_guard<std::mutex> guard(lock_);
        const auto found = std::find_if(surfaces_.begin(), surfaces_.end(), [=](const Resource& r) { return r.Get() == surface; });
        if (found == surfaces_.end()) return 0;
        if (auto_count_ && !auto_last_ && frame_number + 1 >= auto_min_) {
            history_[history_next_] = {list, surface, frame_number + 1, attachment, operation, key};
            history_next_ = (history_next_ + 1) % history_limit;
            if (history_count_ < history_limit) ++history_count_;
        }
        if (!interested()) return 0;
        if (entries_.size() - uav_samples_ >= 512) {
            if (!full_logged_) log("x2w-red producer capture limit reached at frame %ld", frame_number + 1);
            full_logged_ = true;
            return 0;
        }
        const auto desc = surface->GetDesc();
        if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1 ||
            desc.DepthOrArraySize != 1 || desc.MipLevels != 1 || desc.Format != DXGI_FORMAT_R8G8B8A8_UNORM) return 0;
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = after_offset + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
        buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        const auto ordinal = static_cast<unsigned>(entries_.size() + 1);
        Entry entry{list, {}, *found, {}, nullptr, frame_number + 1, 0, attachment, ordinal, 0, operation, key};
        const HRESULT hr = device_of(the_device)->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&entry.readback));
        if (FAILED(hr)) {
            log("x2w-red producer readback allocation failed hr=0x%08lx", static_cast<unsigned long>(hr));
            return 0;
        }
        x2w_budget::committed(device_of(the_device), entry.readback.Get(), D3D12_HEAP_TYPE_READBACK);
        entries_.push_back(std::move(entry));
        copy_pixel(owner, entries_.back(), 0);
        return ordinal;
    }

    bool pair_present(LONG frame, ID3D12CommandQueue* queue)
    {
        if (!pair_) return false;
        std::lock_guard<std::mutex> guard(lock_);
        return std::any_of(entries_.begin(), entries_.end(), [=](const Entry& e) {
            return e.bytes && !e.ambiguous && e.queue == queue && e.record_frame == frame && e.submit_frame == frame;
        });
    }

    int uav_slot(u64 key) const { return uav_.match(key); }

    void after_uav(Wrapper* owner, ID3D12GraphicsCommandList* list, ID3D12Resource* texture, LONG frame, u64 key, unsigned slot)
    {
        std::lock_guard<std::mutex> guard(lock_);
        if (!frame || trace_uav_busy_frame != frame || uav_samples_ >= 2 || uav_last_frame_ == frame) return;
        const auto& states = pass_of(owner).states;
        const auto* bound = states.find(texture);
        if (!bound || bound->depth != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) return;
        const auto desc = texture->GetDesc();
        if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.DepthOrArraySize != 1 || desc.SampleDesc.Count != 1) return;
        Entry entry{list, {}, Resource(texture), {}, nullptr, frame, 0, slot, static_cast<unsigned>(entries_.size() + 1), 0, "compute-uav", key};
        auto* device = device_of(the_device);
        device->GetCopyableFootprints(&desc, 0, 1, 0, &entry.footprint, &entry.rows, &entry.row_bytes, &entry.bytes);
        if (!entry.bytes || entry.bytes > (32ull << 20) || entry.rows != desc.Height || !desc.Width ||
            entry.row_bytes % desc.Width || entry.row_bytes / desc.Width > 16) return;
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = entry.bytes;
        buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST,
                                                   nullptr, IID_PPV_ARGS(&entry.readback)))) return;
        x2w_budget::committed(device, entry.readback.Get(), D3D12_HEAP_TYPE_READBACK);
        const D3D12_TEXTURE_COPY_LOCATION from{texture, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        D3D12_TEXTURE_COPY_LOCATION to{entry.readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        to.PlacedFootprint = entry.footprint;
        require_state(owner, texture, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        require_state(owner, texture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        entries_.push_back(std::move(entry));
        uav_last_frame_ = frame;
        ++uav_samples_;
    }

    void after(Wrapper* owner, unsigned ordinal)
    {
        if (!ordinal) return;
        std::lock_guard<std::mutex> guard(lock_);
        copy_pixel(owner, entries_[ordinal - 1], after_offset);
    }

private:
    void copy_pixel(Wrapper* owner, const Entry& entry, UINT64 offset)
    {
        ID3D12Resource* surface = entry.surface.Get();
        const auto desc = surface->GetDesc();
        D3D12_TEXTURE_COPY_LOCATION to{entry.readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        to.PlacedFootprint.Offset = offset;
        to.PlacedFootprint.Footprint = {desc.Format, 1, 1, 1, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT};
        const D3D12_TEXTURE_COPY_LOCATION from{surface, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        const UINT x = static_cast<UINT>(desc.Width / 2);
        const UINT y = x2w_present_signature::sampled_row(x2w_present_signature::cells_y / 2, desc.Height, x2w_present_signature::cells_y);
        const D3D12_BOX pixel{x, y, 0, x + 1, y + 1, 1};
        require_state(owner, surface, D3D12_RESOURCE_STATE_COPY_SOURCE);
        entry.list->CopyTextureRegion(&to, 0, 0, 0, &from, &pixel);
        require_state(owner, surface, D3D12_RESOURCE_STATE_RENDER_TARGET);
    }

public:
    void discard(ID3D12GraphicsCommandList* list)
    {
        if (!enabled()) return;
        std::lock_guard<std::mutex> guard(lock_);
        for (Entry& entry : entries_) if (entry.list == list) entry.live = false;
    }

    template <class Invoke>
    void execute(ID3D12CommandQueue* queue, const std::vector<ID3D12CommandList*>& lists, const Invoke& invoke)
    {
        if (!enabled()) { invoke(); return; }
        std::lock_guard<std::mutex> guard(lock_);
        flush_locked();
        invoke();
        Fence fence;
        bool tried = false;
        for (ID3D12CommandList* list : lists) {
            for (Entry& entry : entries_) {
                if (!entry.live || entry.list != list) continue;
                if (!tried) {
                    tried = true;
                    if (FAILED(device_of(the_device)->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))) ||
                        FAILED(queue->Signal(fence.Get(), 1))) {
                        log("x2w-red producer fence failed; keeping recording buffers alive");
                        fence.Reset();
                    }
                }
                entry.ambiguous |= entry.pending || !fence;
                entry.fence = fence;
                entry.queue = queue;
                entry.submit_frame = frame_number + 1;
                entry.submit_order = ++submit_order_;
                entry.pending = true;
            }
        }
    }

    void flush()
    {
        if (!enabled()) return;
        std::lock_guard<std::mutex> guard(lock_);
        flush_locked();
    }

private:
    void flush_locked()
    {
        for (Entry& entry : entries_) {
            if (!entry.pending || !entry.fence || entry.fence->GetCompletedValue() != 1) continue;
            entry.pending = false;
            if (entry.ambiguous) {
                log("x2w-red producer ambiguous resubmit ordinal=%u order=%u", entry.ordinal, entry.submit_order);
                continue;
            }
            if (entry.bytes) {
                const D3D12_RANGE read{0, static_cast<SIZE_T>(entry.bytes)}, none{};
                void* memory = nullptr;
                if (SUCCEEDED(entry.readback->Map(0, &read, &memory))) {
                    char label[192];
                    std::snprintf(label, sizeof(label), "x2w-frame post-dispatch %016llx u%u frame %ld order %u target %p",
                                  entry.key, entry.attachment, entry.record_frame, entry.submit_order, static_cast<void*>(entry.surface.Get()));
                    report_texture_pixels(label, label, memory, entry.surface->GetDesc(), entry.footprint, entry.rows, entry.row_bytes);
                    if (pair_ && entry.record_frame == entry.submit_frame) report_frame_rgb("lighting-uav", entry.submit_frame, entry.surface.Get(), memory,
                                                entry.surface->GetDesc(), entry.footprint);
                    entry.readback->Unmap(0, &none);
                }
                continue;
            }
            const D3D12_RANGE read{0, after_offset + 4}, none{};
            void* memory = nullptr;
            if (FAILED(entry.readback->Map(0, &read, &memory))) continue;
            unsigned char rgba[4], prior[4];
            std::memcpy(prior, memory, sizeof(prior));
            std::memcpy(rgba, static_cast<unsigned char*>(memory) + after_offset, sizeof(rgba));
            entry.readback->Unmap(0, &none);
            log("x2w-red producer order=%u ordinal=%u record=%ld submit=%ld queue=%p target=%p op=%s key=%016llx rt=%u rgba=%u,%u,%u,%u before=%u,%u,%u,%u",
                entry.submit_order, entry.ordinal, entry.record_frame, entry.submit_frame, entry.queue, entry.surface.Get(),
                entry.operation, entry.key, entry.attachment, rgba[0], rgba[1], rgba[2], rgba[3],
                prior[0], prior[1], prior[2], prior[3]);
        }
    }
};

RedProducers& red_producers()
{
    static RedProducers disabled(RedProducers::Disabled{});
    // Closed lists may be submitted again and still hold these GPU references.
    static auto* const tracker = [] {
        char value[2]{};
        if (GetEnvironmentVariableA("X2W_CX_TRACE_UAV", value, sizeof(value)) ||
            GetEnvironmentVariableA("X2W_RED_PRODUCER_FRAMES", value, sizeof(value)) ||
            GetEnvironmentVariableA("X2W_RED_PRODUCER_AUTO", value, sizeof(value))) return new RedProducers;
        return &disabled;
    }();
    return *tracker;
}

struct RedDrawSamples {
    unsigned marks[8]{};
    unsigned count = 0;
};

RedDrawSamples red_producer_draw_before(Wrapper* self, ID3D12GraphicsCommandList* list, const char* operation, u64 key)
{
    RedDrawSamples samples;
    auto& tracker = red_producers();
    if (!tracker.may_record()) return samples;
    const Pass& pass = pass_of(self);
    if (!pass.targets_bound) return samples;
    if (tracker.interested())
        log("x2w-red attachments record=%ld list=%p op=%s key=%016llx dsv=%p/%d targets=%u rt0=%p/%d rt1=%p/%d rt2=%p/%d rt3=%p/%d rt4=%p/%d rt5=%p/%d rt6=%p/%d rt7=%p/%d",
            frame_number + 1, list, operation, key, pass.trace_depth, pass.dsv_format, pass.target_count,
            pass.trace_targets[0], pass.target_formats[0], pass.trace_targets[1], pass.target_formats[1],
            pass.trace_targets[2], pass.target_formats[2], pass.trace_targets[3], pass.target_formats[3],
            pass.trace_targets[4], pass.target_formats[4], pass.trace_targets[5], pass.target_formats[5],
            pass.trace_targets[6], pass.target_formats[6], pass.trace_targets[7], pass.target_formats[7]);
    for (UINT i = 0; i < pass.target_count; ++i) {
        if (const unsigned mark = tracker.before(self, list, pass.trace_targets[i], operation, key, i))
            samples.marks[samples.count++] = mark;
    }
    return samples;
}

void red_producer_draw_after(Wrapper* self, const RedDrawSamples& samples)
{
    for (unsigned i = 0; i < samples.count; ++i) red_producers().after(self, samples.marks[i]);
}
