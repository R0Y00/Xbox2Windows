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
};

struct MeshXMemRecording {
    struct Write { Wrapper* owner; u64 offset, bytes; bool leased; };
    std::vector<Write> writes;
    std::shared_ptr<MeshXMemLease> reader, writer;
    u64 reader_address{};
};
struct MeshXMemFlight {
    std::shared_ptr<MeshXMemLease> lease;
    x2w_startup::ComPtr<ID3D12Fence> completion;
};

std::mutex mesh_xmem_mutex;
std::weak_ptr<MeshXMemLease> active_mesh_lease;
std::shared_ptr<MeshXMemLease> published_mesh_lease;
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
    if (published_mesh_lease && published_mesh_lease->owner == w)
        published_mesh_lease.reset();
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
                                                   u64 offset, u64 address, u64 bytes)
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
    result->view = MapViewOfFile(lease.section, FILE_MAP_WRITE, static_cast<DWORD>(lease.section_offset >> 32),
                                 static_cast<DWORD>(lease.section_offset), static_cast<SIZE_T>(lease.window_bytes));
    CloseHandle(lease.section);
    lease.section = nullptr;
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
        if (!source || offset > UINT64_MAX - base || recorded.reader || (base + offset) % alignment) return;
        x2w_startup::ComPtr<ID3D12Device> source_device;
        if (FAILED(source->GetDevice(IID_PPV_ARGS(&source_device))) ||
            source_device.Get() != device_of(list->own().list_device)) return;
        const u64 address = base + offset;
        const auto placed = placed_buffer_range(list->own().list_device, address, bytes);
        if (placed.reject || placed.owner != owner) return;
        auto lease = active_mesh_lease.lock();
        if (!lease) {
            lease = open_mesh_xmem_lease(list->own().list_device, owner, offset, address, bytes);
            if (!lease) return;
            active_mesh_lease = lease;
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
        published_mesh_lease.reset();
        log("x2w-render mesh XMem handoff refused: recording allocation failed");
    }
}

const char* record_mesh_xmem_read(Wrapper* list, u64 address, u64 bytes, D3D12_RESOURCE_STATES state,
                                  ID3D12Resource*& resource, u64& offset)
{
    const std::lock_guard lock(mesh_xmem_mutex);
    const auto lease = published_mesh_lease;
    if (mesh_xmem_tracking_failed || !lease || lease->device != list->own().list_device || lease->bytes != bytes)
        return "mesh XMem read has no submitted writer for the required size";
    const u64 alignment = bytes == sizeof(UINT) ? D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT : sizeof(UINT);
    if (!address || address % alignment)
        return "mesh XMem read address is missing or misaligned";
    const auto placed = placed_buffer_range(list->own().list_device, address, bytes);
    if (!lease->matches(address) || placed.reject || placed.owner != lease->owner)
        return "mesh XMem read does not match the writer's leased page and byte range";
    auto* desktop_list = static_cast<ID3D12GraphicsCommandList*>(list->own().real);
    try {
        auto& recorded = mesh_xmem_recordings[desktop_list];
        if (recorded.writer || (recorded.reader && (recorded.reader != lease || recorded.reader_address != address)))
            return "mesh XMem read requires a separate recording from its writer";
        for (const auto& write : recorded.writes)
            if (write.owner == lease->owner && write.offset < lease->offset + bytes &&
                lease->offset < write.offset + write.bytes)
                return "mesh XMem read overlaps a write in the same recording";
        pass_of(list).states.reserve(pass_of(list).states.size() + 1);
        require_state(list, lease->buffer.Get(), state);
        recorded.reader = lease;
        recorded.reader_address = address;
        resource = lease->buffer.Get();
        offset = lease->lease.first_byte;
        static LONG reports;
        if (first_calls(reports, 16))
            log("x2w-render mesh XMem %s: lease %llu section page +0x%llx byte +0x%llx, %llu bytes",
                state == D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT ? "indirect bind" : "root bind",
                lease->lease.id, lease->lease.section_offset, lease->lease.first_byte, bytes);
    } catch (const std::bad_alloc&) {
        return "mesh XMem read recording allocation failed";
    }
    return nullptr;
}

const char* bind_mesh_xmem_srv(Wrapper* list, UINT parameter, u64 address)
{
    ID3D12Resource* resource{};
    u64 offset{};
    if (const char* failure = record_mesh_xmem_read(list, address, sizeof(UINT),
                                                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, resource, offset))
        return failure;
    auto* desktop_list = static_cast<ID3D12GraphicsCommandList*>(list->own().real);
    desktop_list->SetGraphicsRootShaderResourceView(parameter, resource->GetGPUVirtualAddress() + offset);
    return nullptr;
}

// Caller holds mesh_xmem_mutex. Other copied placements can keep their
// original batching when they cannot change the single active publication.
bool mesh_xmem_needs_submission(ID3D12CommandList* list)
{
    const auto found = mesh_xmem_recordings.find(list);
    if (found == mesh_xmem_recordings.end()) return false;
    const auto& work = found->second;
    const auto active = active_mesh_lease.lock();
    return work.reader || work.writer || (active && std::any_of(work.writes.begin(), work.writes.end(),
        [&](const MeshXMemRecording::Write& write) { return write.owner == active->owner; }));
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
    const auto lease = work.reader ? work.reader : work.writer;
    if (lease) {
        const u64 address = work.reader ? work.reader_address : lease->address;
        const auto placed = placed_buffer_range(lease->device, address, lease->bytes);
        if (mesh_xmem_tracking_failed || lease->untracked || placed.reject || placed.owner != lease->owner ||
            !lease->matches(address) || (work.reader && published_mesh_lease != lease)) {
            log("x2w-render mesh XMem submit refused: writer or live page identity changed");
            return;
        }
    }

    x2w_startup::ComPtr<ID3D12Fence> completion;
    if (lease) {
        x2w_startup::ComPtr<ID3D12Device> device;
        if (submit_fence_fails() || FAILED(queue->GetDevice(IID_PPV_ARGS(&device))) ||
            device.Get() != device_of(lease->device) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&completion)))) {
            log("x2w-render mesh XMem submit refused: no completion fence on this device");
            return;
        }
        try {
            mesh_xmem_flights.reserve(mesh_xmem_flights.size() + 1);
        } catch (const std::bad_alloc&) {
            log("x2w-render mesh XMem submit refused: retirement allocation failed");
            return;
        }
        if (lease->last_access && lease->last_queue != queue) {
            const bool skip = work.reader && x2w_local::env_on("X2W_TEST_MESH_XMEM_SKIP_WAIT");
            if (!skip && FAILED(queue->Wait(lease->last_access.Get(), 1))) {
                log("x2w-render mesh XMem submit refused: cross-queue wait failed");
                return;
            }
            if (!skip) {
                static LONG reports;
                if (first_calls(reports, 16)) log("x2w-render mesh XMem GPU wait for lease %llu", lease->lease.id);
            }
        }
    }
    execute();
    if (lease) {
        if (FAILED(queue->Signal(completion.Get(), 1))) {
            lease->untracked = true;
            published_mesh_lease.reset();
            mesh_xmem_flights.push_back({lease, {}});
            log("x2w-render mesh XMem completion signal failed; lease remains pinned");
            return;
        }
        mesh_xmem_flights.push_back({lease, completion});
        lease->last_access = completion;
        lease->last_queue = queue;
        watch_mesh_xmem_completion(completion.Get());
    }
    for (const auto& write : work.writes) {
        if (write.leased && InterlockedCompareExchange(&lease->owner->own().refs, 0, 0)) {
            published_mesh_lease = lease;
            static LONG reports;
            if (first_calls(reports, 16))
                log("x2w-render mesh XMem writer: lease %llu section page +0x%llx byte +0x%llx, %llu bytes",
                    lease->lease.id, lease->lease.section_offset, lease->lease.first_byte, lease->bytes);
        } else if (published_mesh_lease && write.owner == published_mesh_lease->owner &&
                   write.offset < published_mesh_lease->offset + published_mesh_lease->bytes &&
                   published_mesh_lease->offset < write.offset + write.bytes) {
            published_mesh_lease.reset();
        }
    }
}
