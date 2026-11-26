// Correlate recorded BC copy destinations with SRVs the binding path actually uses.
// Store identities only. The trace never holds a resource alive or selects a descriptor.
namespace bc_bind_trace {
bool enabled()
{
    static const bool on = x2w_local::env_on("X2W_GPU_BC_BIND_TRACE");
    return on;
}

struct Writer {
    ID3D12Resource* host{};
    const Wrapper* wrapper{};
    const Wrapper* list{};
    u64 address{}, span{};
    D3D12_RESOURCE_DESC desc{};
    UINT sub{};
    LONG frame{};
    bool submitted{};
};
struct Match {
    ID3D12Resource* writer{};
    ID3D12Resource* reader{};
    u64 write_address{}, read_address{};
    bool submitted{};
    bool operator==(const Match& other) const {
        return writer == other.writer && reader == other.reader && write_address == other.write_address &&
               read_address == other.read_address && submitted == other.submitted;
    }
};

std::mutex mutex;
std::vector<Writer> writers;
std::vector<Match> same, different;
std::vector<std::pair<const Wrapper*, size_t>> pending;
std::atomic<u64> generation{1};
constexpr size_t limit = 128;

void copied(Wrapper* list, Wrapper* wrapper, ID3D12Resource* host, UINT sub)
{
    const auto* info = resource_info(wrapper);
    if (!info) return;
    const auto address = reinterpret_cast<u64>(wrapper->own().title_memory);
    const auto& d = info->console;
    const auto layout = detile_layout(d);
    // An exact base still matches when the title layout has no known footprint.
    const auto span = layout.valid ? layout.size : 1;
    std::lock_guard guard(mutex);
    size_t index = 0;
    while (index < writers.size() && !(writers[index].host == host && writers[index].address == address &&
                                       writers[index].wrapper == wrapper)) ++index;
    if (index == limit) return;
    if ((index == writers.size() || !writers[index].submitted) && pending.size() < 4 * limit) {
        const auto key = std::make_pair(static_cast<const Wrapper*>(list), index);
        if (std::find(pending.begin(), pending.end(), key) == pending.end()) pending.push_back(key);
    }
    if (index < writers.size()) return;

    writers.push_back({host, wrapper, list, address, span, d, sub, frame_number, false});
    generation.fetch_add(1, std::memory_order_release);
    log("x2w-bc copied frame=%ld list=%p title=0x%llx host=%p wrapper=%p span=%llu sub=%u "
        "size=%llux%ux%u mips=%u fmt=%u layout=0x%x flags=0x%x",
        frame_number, list, address, host, wrapper, span, sub, d.Width, d.Height, d.DepthOrArraySize,
        d.MipLevels, unsigned(d.Format), unsigned(d.Layout), unsigned(d.Flags));
}

void close_recording(const Wrapper* list, bool submitted)
{
    std::lock_guard guard(mutex);
    pending.erase(std::remove_if(pending.begin(), pending.end(), [&](const auto& record) {
        if (record.first != list) return false;
        auto& w = writers[record.second];
        if (submitted && !w.submitted) {
            w.submitted = true;
            generation.fetch_add(1, std::memory_order_release);
            log("x2w-bc submitted frame=%ld list=%p title=0x%llx host=%p", frame_number, list, w.address, w.host);
        }
        return true;
    }), pending.end());
}
void read(const Wrapper* list, const Wrapper* resource, ID3D12Resource* host, const D3D12_RESOURCE_DESC& rd)
{
    if (!resource || resource->own().kind != resource_kind) return;
    const auto address = reinterpret_cast<u64>(resource->own().title_memory);
    struct Recent { u64 generation{}, address{}; ID3D12Resource* host{}; };
    static thread_local std::array<Recent, 256> recent{};
    auto& entry = recent[((address >> 16) ^ (reinterpret_cast<u64>(host) >> 4)) % recent.size()];
    const auto epoch = generation.load(std::memory_order_acquire);
    if (entry.generation == epoch && entry.host == host && entry.address == address) return;
    entry = {epoch, address, host};
    std::lock_guard guard(mutex);
    for (const auto& w : writers) {
        const bool identical = w.host == host;
        if (!identical && (!w.address || address < w.address || address - w.address >= w.span)) continue;
        auto& seen = identical ? same : different;
        const Match match{w.host, host, w.address, address, w.submitted};
        if (seen.size() == limit || std::find(seen.begin(), seen.end(), match) != seen.end()) continue;
        seen.push_back(match);
        const auto* info = resource_info(resource);
        const auto& d = info ? info->console : rd;
        log("x2w-bc bound frame=%ld list=%p same_host=%u submitted=%u copied_frame=%ld write_title=0x%llx "
            "write_host=%p read_title=0x%llx read_host=%p wrapper=%p size=%llux%ux%u mips=%u fmt=%u layout=0x%x "
            "flags=0x%x proxy=%u detiled=%u partial=%u",
            frame_number, list, unsigned(identical), unsigned(w.submitted), w.frame, w.address, w.host, address, host,
            resource, d.Width, d.Height, d.DepthOrArraySize, d.MipLevels, unsigned(d.Format), unsigned(d.Layout),
            unsigned(d.Flags), unsigned(resource->own().linear_memory_srv), unsigned(resource->own().detiled_memory_srv),
            unsigned(resource->own().partial_mip_srv));
    }
}
} // namespace bc_bind_trace
