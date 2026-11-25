// Included after the command-list resource state tracker. Only slot 15 can
// publish this experimental lease. Other consumers of the placed resource
// keep using its normal desktop backing.
bool mesh_xmem_handoff()
{
    static const bool enabled = x2w_local::env_on("X2W_MESH86_XMEM_SRV");
    return enabled;
}

bool mesh_xmem_indirect()
{
    static const bool enabled = x2w_local::env_on("X2W_MESH_INDIRECT_XMEM");
    return enabled;
}

bool mesh_xmem_tracking()
{
    return mesh_xmem_handoff() || mesh_xmem_indirect();
}

enum MeshXMemRange { mesh_srv_range, mesh_argument_range, mesh_range_count };

MeshXMemRange mesh_xmem_range(u64 bytes)
{
    return bytes == sizeof(UINT) ? mesh_srv_range : mesh_argument_range;
}

struct MeshXMemLease {
    using Release = BOOL (WINAPI*)(UINT, ULONG64);
    X2WXMemSectionLease lease{};
    Release release{};
    X2WXMemMatchSection match{};
    Wrapper* owner{};
    Wrapper* device{};
    u64 offset{}, address{}, bytes{};
    void* view{};
    x2w_startup::ComPtr<ID3D12Heap> heap;
    x2w_startup::ComPtr<ID3D12Resource> buffer;
    x2w_startup::ComPtr<ID3D12Fence> last_access;
    ID3D12CommandQueue* last_queue{};
    bool untracked{};

    ~MeshXMemLease()
    {
        buffer.Reset();
        heap.Reset();
        if (lease.section) CloseHandle(lease.section);
        // A view that could not be removed still has access to the slots.
        if ((!view || UnmapViewOfFile(view)) && lease.id)
            release(x2w_xmem_section_lease_version, lease.id);
    }

    bool matches(u64 candidate) const
    {
        return match(x2w_xmem_section_lease_version, lease.id, lease.first_byte, candidate, bytes);
    }

    bool overlaps_pages(const MeshXMemLease& other) const
    {
        const auto& rhs = other.lease;
        if (lease.section_offset >= rhs.section_offset + rhs.window_bytes ||
            rhs.section_offset >= lease.section_offset + lease.window_bytes) return false;
        using Compare = BOOL (WINAPI*)(HANDLE, HANDLE);
        static const auto compare = reinterpret_cast<Compare>(
            GetProcAddress(GetModuleHandleW(L"kernelbase.dll"), "CompareObjectHandles"));
        return !compare || compare(lease.section, rhs.section);
    }
};

using MeshXMemLeases = std::array<std::shared_ptr<MeshXMemLease>, mesh_range_count>;

struct MeshXMemRead {
    u64 address{}, bytes{};
    D3D12_RESOURCE_STATES state{};
    ID3D12Resource* resource{};
    u64 offset{};
};

struct MeshXMemRecording {
    struct Write { Wrapper* owner; u64 offset, bytes; bool leased; };
    std::vector<Write> writes;
    MeshXMemLeases readers;
    std::array<u64, mesh_range_count> reader_addresses{};
    std::shared_ptr<MeshXMemLease> writer;

    bool has_reader() const { return readers[mesh_srv_range] || readers[mesh_argument_range]; }
};
struct MeshXMemFlight {
    std::shared_ptr<MeshXMemLease> lease;
    x2w_startup::ComPtr<ID3D12Fence> completion;
};

std::mutex mesh_xmem_mutex;
std::array<std::weak_ptr<MeshXMemLease>, mesh_range_count> active_mesh_leases;
MeshXMemLeases published_mesh_leases;
std::unordered_map<ID3D12CommandList*, MeshXMemRecording> mesh_xmem_recordings;
std::vector<MeshXMemFlight> mesh_xmem_flights;
bool mesh_xmem_tracking_failed;

// All functions that mutate publication/flight state hold mesh_xmem_mutex.
void reap_mesh_xmem_flights()
{
    mesh_xmem_flights.erase(std::remove_if(mesh_xmem_flights.begin(), mesh_xmem_flights.end(),
        [](const MeshXMemFlight& flight) {
            return flight.completion && flight.completion->GetCompletedValue() == 1;
        }), mesh_xmem_flights.end());
}

void CALLBACK mesh_xmem_completed(PTP_CALLBACK_INSTANCE, void* context, PTP_WAIT wait, TP_WAIT_RESULT)
{
    {
        const std::lock_guard lock(mesh_xmem_mutex);
        reap_mesh_xmem_flights();
    }
    CloseThreadpoolWait(wait);
    CloseHandle(static_cast<HANDLE>(context));
}

void watch_mesh_xmem_completion(ID3D12Fence* fence)
{
    // The callback may outlive the last application reference to the shim.
    static const bool module_pinned = [] {
        HMODULE module{};
        return GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                                  reinterpret_cast<LPCWSTR>(&mesh_xmem_completed), &module) != FALSE;
    }();
    if (!module_pinned) return;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return;
    const auto wait = CreateThreadpoolWait(mesh_xmem_completed, event, nullptr);
    if (!wait || FAILED(fence->SetEventOnCompletion(1, event))) {
        if (wait) CloseThreadpoolWait(wait);
        CloseHandle(event);
        // Later wrapper activity can still reap the recorded fence.
        return;
    }
    SetThreadpoolWait(wait, event, nullptr);
}

void drop_mesh_lease_owner(Wrapper* w)
{
    if (!mesh_xmem_tracking()) return;
    const std::lock_guard lock(mesh_xmem_mutex);
    for (auto& published : published_mesh_leases)
        if (published && published->owner == w) published.reset();
    reap_mesh_xmem_flights();
}

void discard_mesh_lease_recording(ID3D12CommandList* list)
{
    if (!mesh_xmem_tracking()) return;
    const std::lock_guard lock(mesh_xmem_mutex);
    mesh_xmem_recordings.erase(list);
    reap_mesh_xmem_flights();
}

std::shared_ptr<MeshXMemLease> open_mesh_xmem_lease(Wrapper* device, Wrapper* owner,
                                                   u64 offset, u64 address, u64 bytes,
                                                   const MeshXMemLease* other)
{
    using Pin = BOOL (WINAPI*)(UINT, ULONG_PTR, SIZE_T, X2WXMemSectionLease*, SIZE_T);
    HMODULE xmem = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN, L"xmem.dll", &xmem)) return {};
    const auto pin = reinterpret_cast<Pin>(GetProcAddress(xmem, "x2w_xmem_lease_section"));
    const auto release = reinterpret_cast<MeshXMemLease::Release>(GetProcAddress(xmem, "x2w_xmem_release_section"));
    const auto match = reinterpret_cast<X2WXMemMatchSection>(GetProcAddress(xmem, "x2w_xmem_match_section"));
    if (!pin || !release || !match) return {};
    D3D12_FEATURE_DATA_EXISTING_HEAPS feature{};
    x2w_startup::ComPtr<ID3D12Device3> desktop;
    if (FAILED(device_of(device)->QueryInterface(IID_PPV_ARGS(&desktop)))) return {};
    const auto check_heaps = [&] {
#if defined(X2W_MESH_XMEM_AGILITY_620) && defined(__ID3D12Device16_INTERFACE_DEFINED__)
        x2w_startup::ComPtr<ID3D12Device16> versioned;
        if (SUCCEEDED(desktop.As(&versioned)))
            return versioned->CheckFeatureSupport1(D3D12_FEATURE_EXISTING_HEAPS, &feature, sizeof(feature), 620);
#endif
        return desktop->CheckFeatureSupport(D3D12_FEATURE_EXISTING_HEAPS, &feature, sizeof(feature));
    };
    if (FAILED(check_heaps()) || !feature.Supported) return {};
    auto result = std::make_shared<MeshXMemLease>();
    result->owner = owner;
    result->device = device;
    result->address = address;
    result->offset = offset;
    result->bytes = bytes;
    result->release = release;
    result->match = match;
    auto& lease = result->lease;
    if (!pin(x2w_xmem_section_lease_version, address, bytes, &lease, sizeof(lease))) return {};
    if (other && result->overlaps_pages(*other)) {
        log("x2w-render mesh XMem writer refused: root SRV and arguments share a physical page");
        return {};
    }
    result->view = MapViewOfFile(lease.section, FILE_MAP_WRITE, static_cast<DWORD>(lease.section_offset >> 32),
                                 static_cast<DWORD>(lease.section_offset), static_cast<SIZE_T>(lease.window_bytes));
    // Keep the section handle for physical page comparison with the other lease.
    if (!result->view || FAILED(desktop->OpenExistingHeapFromAddress(result->view, IID_PPV_ARGS(&result->heap))))
        return {};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = lease.window_bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
    if (FAILED(desktop->CreatePlacedResource(result->heap.Get(), 0, &desc, D3D12_RESOURCE_STATE_COMMON,
                                             nullptr, IID_PPV_ARGS(&result->buffer)))) return {};
    return result->matches(address) ? result : nullptr;
}

// Track other writes to the copied placement too. An overlapping write that
// did not update the XMem view invalidates its publication at submission.
void note_mesh_xmem_write(Wrapper* list, Wrapper* owner, u64 offset, u64 bytes,
                          ID3D12Resource* source = nullptr, u64 source_offset = 0)
{
    if (!mesh_xmem_tracking() || !owner || !owner->own().title_memory) return;
    const auto* info = resource_info(owner);
    if (!info || info->placement != Placement::copied_buffer ||
        info->placement_device != list->own().list_device || info->console.Flags != D3D12_RESOURCE_FLAG_NONE)
        return;
    auto* desktop_list = static_cast<ID3D12GraphicsCommandList*>(list->own().real);
    const std::lock_guard lock(mesh_xmem_mutex);
    if (mesh_xmem_tracking_failed) return;
    try {
        auto& recorded = mesh_xmem_recordings[desktop_list];
        recorded.writes.push_back({owner, offset, bytes, false});
        const u64 base = reinterpret_cast<u64>(owner->own().title_memory);
        const u64 alignment = bytes == sizeof(UINT) ? D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT : sizeof(UINT);
        if (!source || offset > UINT64_MAX - base || recorded.has_reader() || (base + offset) % alignment) return;
        const auto range = mesh_xmem_range(bytes);
        if (recorded.writer && recorded.writer->bytes != bytes) return;
        x2w_startup::ComPtr<ID3D12Device> source_device;
        if (FAILED(source->GetDevice(IID_PPV_ARGS(&source_device))) ||
            source_device.Get() != device_of(list->own().list_device)) return;
        const u64 address = base + offset;
        const auto placed = placed_buffer_range(list->own().list_device, address, bytes);
        if (placed.reject || placed.owner != owner) return;
        auto& active = active_mesh_leases[range];
        auto lease = active.lock();
        if (!lease) {
            const auto other = active_mesh_leases[range == mesh_srv_range ? mesh_argument_range : mesh_srv_range].lock();
            lease = open_mesh_xmem_lease(list->own().list_device, owner, offset, address, bytes, other.get());
            if (!lease) return;
            active = lease;
        }
        if (lease->untracked || lease->owner != owner || lease->offset != offset || lease->bytes != bytes ||
            !lease->matches(address)) return;
        auto& states = pass_of(list).states;
        // Reserve before recording a barrier so a failed allocation cannot
        // leave an untracked transition on the desktop list.
        states.reserve(states.size() + 1);
        require_state(list, lease->buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
        desktop_list->CopyBufferRegion(lease->buffer.Get(), lease->lease.first_byte, source, source_offset, bytes);
        recorded.writer = lease;
        recorded.writes.back().leased = true;
    } catch (const std::bad_alloc&) {
        mesh_xmem_tracking_failed = true;
        published_mesh_leases = {};
        log("x2w-render mesh XMem handoff refused: recording allocation failed");
    }
}

const char* record_mesh_xmem_reads(Wrapper* list, std::array<MeshXMemRead, mesh_range_count>& reads)
{
    const std::lock_guard lock(mesh_xmem_mutex);
    auto* desktop_list = static_cast<ID3D12GraphicsCommandList*>(list->own().real);
    try {
        auto& recorded = mesh_xmem_recordings[desktop_list];
        // Validate the entire command before changing either binding or adding
        // dependencies to the recording.
        for (std::size_t range = 0; range < reads.size(); ++range) {
            const auto& read = reads[range];
            if (!read.bytes) continue;
            const auto& lease = published_mesh_leases[range];
            if (mesh_xmem_tracking_failed || !lease || lease->device != list->own().list_device ||
                lease->bytes != read.bytes)
                return "mesh XMem read has no submitted writer for the required size";
            const u64 alignment = range == mesh_srv_range ? D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT : sizeof(UINT);
            if (!read.address || read.address % alignment)
                return "mesh XMem read address is missing or misaligned";
            const auto placed = placed_buffer_range(list->own().list_device, read.address, read.bytes);
            if (!lease->matches(read.address) || placed.reject || placed.owner != lease->owner)
                return "mesh XMem read does not match the writer's leased page and byte range";
            if (recorded.writer || (recorded.readers[range] &&
                (recorded.readers[range] != lease || recorded.reader_addresses[range] != read.address)))
                return "mesh XMem read requires a separate recording from its writer";
            for (const auto& write : recorded.writes)
                if (write.owner == lease->owner && write.offset < lease->offset + read.bytes &&
                    lease->offset < write.offset + write.bytes)
                    return "mesh XMem read overlaps a write in the same recording";
        }
        auto& states = pass_of(list).states;
        states.reserve(states.size() + reads.size());
        for (std::size_t range = 0; range < reads.size(); ++range) {
            auto& read = reads[range];
            if (!read.bytes) continue;
            const auto& lease = published_mesh_leases[range];
            require_state(list, lease->buffer.Get(), read.state);
            recorded.readers[range] = lease;
            recorded.reader_addresses[range] = read.address;
            read.resource = lease->buffer.Get();
            read.offset = lease->lease.first_byte;
            static LONG reports;
            if (first_calls(reports, 16))
                log("x2w-render mesh XMem %s: lease %llu section page +0x%llx byte +0x%llx, %llu bytes",
                    range == mesh_argument_range ? "indirect bind" : "root bind",
                    lease->lease.id, lease->lease.section_offset, read.offset, read.bytes);
        }
    } catch (const std::bad_alloc&) {
        return "mesh XMem read recording allocation failed";
    }
    return nullptr;
}

// Caller holds mesh_xmem_mutex. Other copied placements can keep their
// original batching when they cannot change either publication.
bool mesh_xmem_needs_submission(ID3D12CommandList* list)
{
    const auto found = mesh_xmem_recordings.find(list);
    if (found == mesh_xmem_recordings.end()) return false;
    const auto& work = found->second;
    if (work.has_reader() || work.writer) return true;
    for (const auto& active : active_mesh_leases)
        if (const auto lease = active.lock())
            if (std::any_of(work.writes.begin(), work.writes.end(),
                [&](const MeshXMemRecording::Write& write) { return write.owner == lease->owner; })) return true;
    return false;
}

// Draw and allocator locks serialize queue submissions. Keep the mesh lock
// through validation, enqueue and publication. No CPU fence wait is taken.
template<class Execute>
void execute_mesh_xmem_recording(ID3D12CommandQueue* queue, ID3D12CommandList* list, Execute execute)
{
    const std::lock_guard lock(mesh_xmem_mutex);
    reap_mesh_xmem_flights();
    const auto record = mesh_xmem_recordings.find(list);
    if (record == mesh_xmem_recordings.end()) {
        execute();
        return;
    }
    const auto& work = record->second;
    auto leases = work.readers;
    if (work.writer) leases[mesh_xmem_range(work.writer->bytes)] = work.writer;
    Wrapper* lease_device = nullptr;
    std::size_t retained = 0;
    for (std::size_t range = 0; range < leases.size(); ++range) {
        const auto& lease = leases[range];
        if (!lease) continue;
        const u64 address = work.reader_addresses[range] ? work.reader_addresses[range] : lease->address;
        const auto placed = placed_buffer_range(lease->device, address, lease->bytes);
        if (mesh_xmem_tracking_failed || lease->untracked || placed.reject || placed.owner != lease->owner ||
            !lease->matches(address) || (work.readers[range] && published_mesh_leases[range] != lease)) {
            log("x2w-render mesh XMem submit refused: writer or live page identity changed (%llu bytes)", lease->bytes);
            return;
        }
        lease_device = lease->device;
        ++retained;
    }

    x2w_startup::ComPtr<ID3D12Fence> completion;
    if (retained) {
        x2w_startup::ComPtr<ID3D12Device> device;
        if (submit_fence_fails() || FAILED(queue->GetDevice(IID_PPV_ARGS(&device))) ||
            device.Get() != device_of(lease_device) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&completion)))) {
            log("x2w-render mesh XMem submit refused: no completion fence on this device");
            return;
        }
        try {
            mesh_xmem_flights.reserve(mesh_xmem_flights.size() + retained);
        } catch (const std::bad_alloc&) {
            log("x2w-render mesh XMem submit refused: retirement allocation failed");
            return;
        }
        ID3D12Fence* waited = nullptr;
        for (std::size_t range = 0; range < leases.size(); ++range) {
            const auto& lease = leases[range];
            if (!lease || !lease->last_access || lease->last_queue == queue || lease->last_access.Get() == waited)
                continue;
            const bool skip = work.readers[range] && (x2w_local::env_on("X2W_TEST_MESH_XMEM_SKIP_WAIT") ||
                (retained == mesh_range_count && x2w_local::env_on(range == mesh_srv_range ?
                    "X2W_TEST_MESH_XMEM_SKIP_SRV_WAIT" : "X2W_TEST_MESH_XMEM_SKIP_ARGUMENT_WAIT")));
            if (skip) {
                log("x2w-render mesh XMem GPU wait omitted for lease %llu, %llu bytes", lease->lease.id, lease->bytes);
                continue;
            }
            if (FAILED(queue->Wait(lease->last_access.Get(), 1))) {
                log("x2w-render mesh XMem submit refused: cross-queue wait failed");
                return;
            }
            waited = lease->last_access.Get();
            static LONG reports;
            if (first_calls(reports, 16))
                log("x2w-render mesh XMem GPU wait for lease %llu, %llu bytes", lease->lease.id, lease->bytes);
        }
    }
    execute();
    if (retained) {
        const bool signaled = SUCCEEDED(queue->Signal(completion.Get(), 1));
        for (std::size_t range = 0; range < leases.size(); ++range) {
            const auto& lease = leases[range];
            if (!lease) continue;
            if (signaled) {
                mesh_xmem_flights.push_back({lease, completion});
                lease->last_access = completion;
                lease->last_queue = queue;
            } else {
                lease->untracked = true;
                if (published_mesh_leases[range] == lease) published_mesh_leases[range].reset();
                mesh_xmem_flights.push_back({lease, {}});
            }
        }
        if (!signaled) {
            log("x2w-render mesh XMem completion signal failed; leases remain pinned");
            return;
        }
        watch_mesh_xmem_completion(completion.Get());
    }
    for (const auto& write : work.writes) {
        if (write.leased && InterlockedCompareExchange(&work.writer->owner->own().refs, 0, 0)) {
            const auto& lease = work.writer;
            published_mesh_leases[mesh_xmem_range(lease->bytes)] = lease;
            static LONG reports;
            if (first_calls(reports, 16))
                log("x2w-render mesh XMem writer: lease %llu section page +0x%llx byte +0x%llx, %llu bytes",
                    lease->lease.id, lease->lease.section_offset, lease->lease.first_byte, lease->bytes);
        } else {
            for (auto& published : published_mesh_leases)
                if (published && write.owner == published->owner && write.offset < published->offset + published->bytes &&
                    published->offset < write.offset + write.bytes) published.reset();
        }
    }
}
