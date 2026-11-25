// Source-built writer and lifetime probes for the cached mesh fixture.
struct MeshLeaseWriter {
    static constexpr UINT64 page = 65536, word_offset = 0x40;
    using Reserve = void* (WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
    using Allocate = BOOL (WINAPI*)(ULONG_PTR, ULONG_PTR*, ULONG_PTR*);
    using Map = BOOL (WINAPI*)(void* const*, ULONG_PTR, const ULONG_PTR*);
    using Free = BOOL (WINAPI*)(ULONG_PTR, ULONG_PTR*);
    using Pin = BOOL (WINAPI*)(UINT, ULONG_PTR, SIZE_T, X2WXMemSectionLease*, SIZE_T);
    using Unpin = BOOL (WINAPI*)(UINT, ULONG64);

    ID3D12Device* device;
    ID3D12Device* guest;
    HMODULE xmem;
    Allocate alloc;
    Map map;
    Free free_pages;
    Pin pin;
    Unpin unpin;
    unsigned char* cpu{};
    UINT64 mirror{};
    ULONG_PTR original[2]{};
    std::pair<ULONG64, ULONG64> published_page;
    ComPtr<IUnknown> placed, recording, queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12Resource> upload;
    ComPtr<ID3D12Fence> gate, completion;
    bool submitted{};

    template<class T> T function(const char* name)
    {
        const auto fn = reinterpret_cast<T>(GetProcAddress(xmem, name));
        check(fn ? S_OK : E_FAIL, name);
        return fn;
    }

    MeshLeaseWriter(ID3D12Device* d, ID3D12Device* g, const wchar_t* dll)
        : device(d), guest(g), xmem(LoadLibraryW(dll)), alloc(function<Allocate>("XMemAllocatePhysicalPages")),
          map(function<Map>("XMemMapPhysicalPagesScatter")), free_pages(function<Free>("XMemFreePhysicalPages")),
          pin(function<Pin>("x2w_xmem_lease_section")), unpin(function<Unpin>("x2w_xmem_release_section"))
    {
        check(SetEnvironmentVariableA("X2W_XMEM_PERMISSIVE", "1") &&
              SetEnvironmentVariableA("X2W_XMEM_GPU_MIRROR", "1") ? S_OK : E_FAIL, "enable XMem mirror");
        const auto reserve = function<Reserve>("XMemVirtualAlloc");
        constexpr UINT64 window = UINT64{16} << 30;
        cpu = static_cast<unsigned char*>(reserve(nullptr, window, 0x20402000, 0x21, 4));
        mirror = reinterpret_cast<UINT64>(reserve(nullptr, window, 0x20402000, 3, 0x801));
        check(cpu && mirror ? S_OK : E_FAIL, "reserve mirrored XMem");
        ULONG_PTR count = 2;
        void* const addresses[]{cpu, cpu + page};
        check(alloc(0x280, &count, original) && count == 2 && map(addresses, 2, original) ? S_OK : E_FAIL,
              "map two XMem pages");
        std::memcpy(cpu + word_offset, &green, sizeof(green));
        std::memcpy(cpu + page + word_offset, &green, sizeof(green));
        published_page = inspect(address());

        const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, page, 1, 1, 1, DXGI_FORMAT_UNKNOWN,
            {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
        using Place = HRESULT (WINAPI*)(IUnknown*, void*, const D3D12_RESOURCE_DESC*, UINT, const void*, REFIID, void**);
        check(slot<Place>(guest, 133)(guest, reinterpret_cast<void*>(mirror), &desc, D3D12_RESOURCE_STATE_COMMON,
                                     nullptr, __uuidof(ID3D12Resource), reinterpret_cast<void**>(placed.GetAddressOf())),
              "create copied XMem placement through slot 133");
        upload = make_buffer(device, sizeof(red), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        const D3D12_RANGE empty{};
        void* mapped = nullptr;
        check(upload->Map(0, &empty, &mapped), "map writer source");
        std::memcpy(mapped, &red, sizeof(red));
        upload->Unmap(0, nullptr);
        check(guest->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&allocator)), "create writer allocator");
        recording = begin_guest_list(guest, allocator.Get(), D3D12_COMMAND_LIST_TYPE_COPY);
        using Copy = void (WINAPI*)(IUnknown*, IUnknown*, UINT64, IUnknown*, UINT64, UINT64);
        slot<Copy>(recording.Get(), 15)(recording.Get(), placed.Get(), word_offset, upload.Get(), 0, sizeof(red));
        check(slot<HRESULT (WINAPI*)(IUnknown*)>(recording.Get(), 9)(recording.Get()), "close slot 15 writer");
        queue = create_queue(D3D12_COMMAND_LIST_TYPE_COPY);
        gate = create_fence();
        completion = create_fence();
    }

    ~MeshLeaseWriter()
    {
        if (submitted) {
            gate->Signal(1);
            completion->SetEventOnCompletion(1, nullptr);
        }
    }

    UINT64 address() const { return mirror + word_offset; }

    ComPtr<IUnknown> create_queue(D3D12_COMMAND_LIST_TYPE type)
    {
        const D3D12_COMMAND_QUEUE_DESC desc{type};
        ComPtr<IUnknown> result;
        check(guest->CreateCommandQueue(&desc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(result.GetAddressOf())),
              "create wrapped lease queue");
        return result;
    }

    ComPtr<ID3D12Fence> create_fence()
    {
        ComPtr<ID3D12Fence> result;
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&result)), "create lease fixture fence");
        return result;
    }

    std::pair<ULONG64, ULONG64> inspect(UINT64 guest_address)
    {
        X2WXMemSectionLease lease{};
        check(pin(x2w_xmem_section_lease_version, guest_address, sizeof(UINT), &lease, sizeof(lease)) ? S_OK : E_FAIL,
              "inspect XMem page");
        CloseHandle(lease.section);
        check(unpin(x2w_xmem_section_lease_version, lease.id) ? S_OK : E_FAIL, "release inspection lease");
        return {lease.section_offset, lease.first_byte};
    }

    void start()
    {
        auto native = native_object<ID3D12CommandQueue>(device, guest, queue.Get());
        check(native->Wait(gate.Get(), 1), "hold writer behind gate");
        submit_wrapped(queue.Get(), recording.Get());
        check(native->Signal(completion.Get(), 1), "signal writer completion");
        submitted = true;
    }

    void finish()
    {
        check(gate->Signal(1), "open writer gate");
        check(completion->SetEventOnCompletion(1, nullptr), "wait for slot 15 writer");
        check(device->GetDeviceRemovedReason(), "slot 15 GPU completion");
        UINT actual{};
        std::memcpy(&actual, cpu + word_offset, sizeof(actual));
        if (actual != red)
            std::fprintf(stderr, "leased slot 15 writer: 0x%08x != 0x%08x\n", actual, red);
        check(actual == red ? S_OK : E_FAIL, "leased XMem writer changed the CPU view");
    }

    void free_original()
    {
        recording.Reset();
        placed.Reset();
        void* const addresses[]{cpu, cpu + page};
        const ULONG_PTR empty[]{0, 0};
        check(map(addresses, 2, empty) && free_pages(2, original) ? S_OK : E_FAIL, "release the original XMem grant");
    }

    std::pair<ULONG64, ULONG64> allocate_probe(UINT64 index)
    {
        ULONG_PTR token{}, count = 1;
        void* const address = cpu + index * page;
        check(alloc(0x280, &count, &token) && count == 1 && map(&address, 1, &token) ? S_OK : E_FAIL, "map probe page");
        const auto identity = inspect(mirror + index * page + word_offset);
        const ULONG_PTR empty{};
        check(map(&address, 1, &empty) && free_pages(1, &token) ? S_OK : E_FAIL, "free probe page");
        return identity;
    }

    void check_retirement(const char* label)
    {
        // The GPU fence is already complete. Allow the runtime's retirement
        // callback to run without issuing more calls to the wrapped queue.
        const auto deadline = GetTickCount64() + 3000;
        do {
            if (allocate_probe(2) == published_page) return;
            Sleep(1);
        } while (GetTickCount64() < deadline);
        check(E_FAIL, label);
    }
};

bool supports_lease_heap(ID3D12Device* device)
{
    D3D12_FEATURE_DATA_EXISTING_HEAPS support{};
    return SUCCEEDED(check_feature(device, D3D12_FEATURE_EXISTING_HEAPS, &support, sizeof(support))) && support.Supported;
}

void check_lease_debug(ID3D12InfoQueue* info)
{
    if (!info) return;
    for (UINT64 i = 0; i < info->GetNumStoredMessages(); ++i) {
        SIZE_T bytes{};
        check(info->GetMessage(i, nullptr, &bytes), "size D3D12 debug message");
        std::vector<unsigned char> storage(bytes);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        check(info->GetMessage(i, message, &bytes), "read D3D12 debug message");
        if (message->Severity <= D3D12_MESSAGE_SEVERITY_WARNING) {
            std::fprintf(stderr, "%s\n", message->pDescription);
            check(E_FAIL, "D3D12 debug layer reported a warning or error");
        }
    }
    std::puts("D3D12 warnings/errors: 0");
}
