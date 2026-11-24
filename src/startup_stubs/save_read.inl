// Synchronous operations on the public GDK save provider identity.
namespace save_read {
struct BlobInfo { const char* name; std::uint32_t size; };
struct Blob { BlobInfo info; unsigned char* data; };
struct ContainerInfo {
    const char* name;
    const char* display_name;
    std::uint32_t blob_count;
    u64 bytes;
    std::time_t modified;
    bool needs_sync;
};
static_assert(sizeof(Blob) == 24 && offsetof(Blob, data) == 16);
static_assert(sizeof(ContainerInfo) == 48 && offsetof(ContainerInfo, bytes) == 24);
constexpr u64 small_buffer = 0x80830007, missing_blob = 0x80830008, missing_container = 0x80830010;

std::string container_name(u64 handle)
{
    if (handle < save_container_handle || handle - save_container_handle >= max_save_handles) return {};
    char local[x2w_save::max_name_bytes + 1]{};
    AcquireSRWLockShared(&save_lock);
    if (const auto* name = save_table.open_name(static_cast<std::size_t>(handle - save_container_handle)))
        std::memcpy(local, name->c_str(), name->size() + 1);
    ReleaseSRWLockShared(&save_lock);
    return local;
}

u64 containers(bool exact, u64* args)
{
    const auto callback = reinterpret_cast<bool(WINAPI*)(const ContainerInfo*, void*)>(args[exact ? 3 : 2]);
    auto* context = reinterpret_cast<void*>(args[exact ? 2 : 1]);
    std::string wanted;
    if (args[0] != save_provider_handle || !callback ||
        (exact && (!read_save_name(args[1], wanted) || !x2w_save::valid_name(wanted)))) return e_invalidarg;
    struct Snapshot { x2w_save::Entry entry; std::uint32_t count; u64 size; };
    std::vector<Snapshot> snapshots;
    const auto store = save_store();
    {
        const std::lock_guard guard(store.publication_lock());
        std::vector<x2w_save::Entry> entries;
        if (auto s = store.entries({}, entries); s != x2w_save::Status::ok) return save_status(s);
        for (const auto& entry : entries) {
            if (exact && _stricmp(entry.name.c_str(), wanted.c_str()) != 0) continue;
            std::vector<x2w_save::Entry> blobs;
            if (auto s = store.entries(entry.name, blobs); s != x2w_save::Status::ok) return save_status(s);
            if (blobs.size() > UINT32_MAX) return e_fail;
            u64 bytes = 0;
            for (const auto& blob : blobs) bytes += blob.bytes;
            snapshots.push_back({entry, static_cast<std::uint32_t>(blobs.size()), bytes});
        }
    }
    if (exact && snapshots.empty()) return missing_container;
    for (const auto& item : snapshots) {
        const char* name = item.entry.name.c_str();
        const ContainerInfo info{name, name, item.count, item.size, item.entry.modified, false};
        if (!callback(&info, context)) break;
    }
    return 0;
}

u64 blobs(u64* args)
{
    const auto name = container_name(args[0]);
    const auto callback = reinterpret_cast<bool(WINAPI*)(const BlobInfo*, void*)>(args[2]);
    if (name.empty() || !callback) return e_invalidarg;
    std::vector<x2w_save::Entry> entries;
    const auto store = save_store();
    {
        const std::lock_guard guard(store.publication_lock());
        if (auto s = store.entries(name, entries); s != x2w_save::Status::ok) return save_status(s);
    }
    for (const auto& entry : entries) {
        const BlobInfo info{entry.name.c_str(), static_cast<std::uint32_t>(entry.bytes)};
        if (!callback(&info, reinterpret_cast<void*>(args[1]))) break;
    }
    return 0;
}

u64 read(u64* args)
{
    const auto name = container_name(args[0]);
    const auto* names = reinterpret_cast<const u64*>(args[1]);
    auto* count = reinterpret_cast<std::uint32_t*>(args[2]);
    const auto capacity = static_cast<std::size_t>(args[3]);
    auto* output = reinterpret_cast<Blob*>(args[4]);
    if (name.empty() || !x2w_local::writable(count, sizeof(*count))) return e_invalidarg;
    const auto requested = *count;
    if (requested > capacity / sizeof(Blob)) return small_buffer;
    if (requested && !x2w_local::readable(names, sizeof(*names) * u64{requested})) return e_invalidarg;
    const auto store = save_store();
    const std::lock_guard guard(store.publication_lock());
    std::vector<x2w_save::Entry> all;
    if (auto s = store.entries(name, all); s != x2w_save::Status::ok) return save_status(s);
    std::vector<const x2w_save::Entry*> selected;
    for (std::uint32_t i = 0; i < requested; ++i) {
        std::string wanted;
        if (!read_save_name(names[i], wanted) || !x2w_save::valid_name(wanted)) return e_invalidarg;
        const auto match = std::find_if(all.begin(), all.end(), [&](const auto& item) {
            return _stricmp(item.name.c_str(), wanted.c_str()) == 0;
        });
        if (match == all.end()) return missing_blob;
        selected.push_back(&*match);
    }
    if (!requested) {
        for (const auto& item : all) selected.push_back(&item);
    }
    if (selected.size() > UINT32_MAX) return e_fail;
    std::size_t bytes = selected.size() * sizeof(Blob);
    for (const auto* item : selected) {
        const auto extra = item->name.size() + 1 + item->bytes;
        if (extra > SIZE_MAX - bytes) return small_buffer;
        bytes += static_cast<std::size_t>(extra);
    }
    if (bytes > capacity) return small_buffer;
    if (bytes && (!x2w_local::writable(output, bytes) || args[4] % alignof(Blob))) return e_invalidarg;
    std::vector<std::vector<unsigned char>> data(selected.size());
    for (std::size_t i = 0; i < selected.size(); ++i) {
        if (auto s = store.read_blob(name, selected[i]->name, data[i]); s != x2w_save::Status::ok) return save_status(s);
        if (data[i].size() != selected[i]->bytes) return e_fail;
    }
    if (!selected.empty()) {
        auto* cursor = reinterpret_cast<unsigned char*>(output + selected.size());
        for (std::size_t i = 0; i < selected.size(); ++i) {
            const auto& entry = *selected[i];
            auto* saved_name = reinterpret_cast<char*>(cursor);
            std::memcpy(cursor, entry.name.c_str(), entry.name.size() + 1);
            cursor += entry.name.size() + 1;
            output[i] = {{saved_name, static_cast<std::uint32_t>(entry.bytes)}, cursor};
            if (!data[i].empty()) std::memcpy(cursor, data[i].data(), data[i].size());
            cursor += data[i].size();
        }
    }
    *count = static_cast<std::uint32_t>(selected.size());
    return 0;
}
}

bool save_read_call(int slot, u64* args, u64* result)
{
    try {
        switch (slot) {
        case 13: case 14: *result = save_read::containers(slot == 13, args); break;
        case 18: *result = save_read::blobs(args); break;
        case 20: *result = save_read::read(args); break;
        default: return false;
        }
    } catch (const std::bad_alloc&) {
        *result = e_outofmemory;
    } catch (const std::length_error&) {
        *result = e_outofmemory;
    } catch (const std::filesystem::filesystem_error&) {
        *result = e_fail;
    }
    return true;
}
