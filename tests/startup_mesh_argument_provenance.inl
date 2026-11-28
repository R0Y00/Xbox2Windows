// Source-built descriptor -> compute writer -> indirect Mesh chain (Refs #3310).
// Included in startup_indirect_draw_indexed.cpp to reuse its real guest wrapper,
// placed-resource helper, Mesh shaders and complete pixel oracle.
bool check_mesh_argument_provenance(ID3D12Device* device, ID3D12Device* guest,
    ID3D12RootSignature* graphics_root, IUnknown* graphics, ID3D12CommandSignature* signature)
{
    char setting[64]{};
    GetEnvironmentVariableA("X2W_MESH_PROVENANCE_CASE", setting, sizeof(setting));
    const std::string scenario = setting;
    const bool missing = scenario == "missing-copy";
    const bool stale = scenario == "stale-copy";
    const bool wrong_srv = scenario == "wrong-srv-offset";
    const bool wrong_argument = scenario == "wrong-argument-offset";
    if (!scenario.empty() && scenario != "positive" && !missing && !stale && !wrong_srv && !wrong_argument)
        return false;
    const char* label = scenario.empty() ? "positive" : scenario.c_str();
    constexpr UINT64 input_offset = 16;
    constexpr UINT64 written_offset = 4;
    const UINT64 consumed_offset = wrong_argument ? 20 : written_offset;
    const UINT expected_x = missing || stale || wrong_srv ? 0 : 2;
    const UINT consumed_x = wrong_argument ? 0 : expected_x;
    const Words seed{0xa5a5a5a5, 0, 1, 1, 0x5a5a5a5a, 0, 1, 1, 0x12345678};
    PlacedBuffer arguments(device, guest, sizeof(seed), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    PlacedBuffer input(device, guest, 64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    const auto upload = make_buffer(device, sizeof(seed), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* memory = nullptr;
    const D3D12_RANGE none{0, 0};
    check(upload->Map(0, &none, &memory), "map provenance seed");
    std::memcpy(memory, seed.data(), sizeof(seed));
    upload->Unmap(0, &none);

    constexpr char producer[] = R"(
ByteAddressBuffer launch : register(t0);
RWByteAddressBuffer data : register(u0);
[numthreads(1,1,1)] void seed_input() { data.Store(16, 2); data.Store(32, 0); }
[numthreads(1,1,1)] void update_args() { data.Store3(4, uint3(launch.Load(0), 1, 1)); }
)";
    D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
    D3D12_ROOT_PARAMETER parameters[2]{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[0].DescriptorTable = {1, &range};
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    const auto writer_root = create_root(device, {2, parameters});
    const auto pipeline = [&](const char* entry) {
        ComPtr<ID3DBlob> code, errors;
        const HRESULT hr = D3DCompile(producer, sizeof(producer) - 1, nullptr, nullptr, nullptr,
            entry, "cs_5_1", 0, 0, &code, &errors);
        if (FAILED(hr) && errors) std::fprintf(stderr, "%s\n", static_cast<const char*>(errors->GetBufferPointer()));
        check(hr, entry);
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = writer_root.Get();
        desc.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        ComPtr<ID3D12PipelineState> result;
        check(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&result)), entry);
        return result;
    };
    const auto seed_pipeline = pipeline("seed_input"), update_pipeline = pipeline("update_args");
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1,
        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ComPtr<ID3D12DescriptorHeap> heap, source_heap;
    check(guest->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)), "create provenance descriptor heap");
    const D3D12_DESCRIPTOR_HEAP_DESC source_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1};
    check(guest->CreateDescriptorHeap(&source_desc, IID_PPV_ARGS(&source_heap)), "create CPU copy-source heap");
    const auto native_heap = native_object<ID3D12DescriptorHeap>(device, guest, heap.Get());
    using Handle = UINT64 (WINAPI*)(IUnknown*, void*);
    const UINT64 guest_cpu = slot<Handle>(source_heap.Get(), 9)(source_heap.Get(), nullptr);
    const UINT64 bound_cpu = slot<Handle>(heap.Get(), 9)(heap.Get(), nullptr);
    const UINT64 desktop_cpu = native_heap->GetCPUDescriptorHandleForHeapStart().ptr;
    using View = void (WINAPI*)(IUnknown*, IUnknown*, const D3D12_SHADER_RESOURCE_VIEW_DESC*, UINT64, UINT64);
    using Copy = void (WINAPI*)(IUnknown*, UINT, UINT64, UINT64, D3D12_DESCRIPTOR_HEAP_TYPE);
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R32_TYPELESS;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.NumElements = 1;
    srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    const auto create_srv = [&](UINT index, UINT64 byte_offset, bool null_resource = false) {
        srv.Buffer.FirstElement = byte_offset / 4;
        slot<View>(guest, 18)(guest, null_resource ? nullptr : input.wrapped.Get(), &srv,
            index ? bound_cpu : guest_cpu, 0);
    };
    // A known null destination makes the missing-copy control deterministic.
    create_srv(1, 0, true);
    create_srv(0, input_offset + 16);
    if (!missing) slot<Copy>(guest, 24)(guest, 1, bound_cpu, guest_cpu,
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    create_srv(0, wrong_srv ? input_offset + 16 : input_offset); // overwrite the source
    if (!missing && !stale) slot<Copy>(guest, 24)(guest, 1, bound_cpu, guest_cpu,
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    std::uint32_t tokens[2][8]{};
    std::memcpy(tokens[0], reinterpret_cast<const void*>(guest_cpu), 32);
    std::memcpy(tokens[1], reinterpret_cast<const void*>(bound_cpu), 32);
    std::printf("provenance descriptor case=%s guest=%llx desktop=%llx source=%llx "
        "input=%p input_offset=%llu token_source=%08x,%u token_bound=%08x,%u\n", label,
        bound_cpu, desktop_cpu, guest_cpu,
        static_cast<void*>(input.native.Get()), wrong_srv || stale ? input_offset + 16 : missing ? 0 : input_offset,
        tokens[0][0], tokens[0][1], tokens[1][0], tokens[1][1]);

    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> completion;
    check(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)), "create provenance DIRECT queue");
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&completion)), "create provenance fence");
    ComPtr<ID3D12Resource> snapshots[2];
    const auto input_snapshot = make_buffer(device, 64, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    Words observed[2]{};
    Words expected = seed;
    expected[1] = expected_x;
    const auto read_record = [&](UINT i) {
        const D3D12_RANGE read{0, sizeof(seed)};
        check(snapshots[i]->Map(0, &read, &memory), "map provenance argument snapshot");
        std::memcpy(observed[i].data(), memory, sizeof(seed));
        snapshots[i]->Unmap(0, &none);
        const UINT index = static_cast<UINT>((i ? consumed_offset : written_offset) / 4);
        std::printf("provenance record phase=%s resource=%p offset=%llu tuple=%u,%u,%u guards=%s\n",
            i ? "pre-consume" : "post-write", static_cast<void*>(arguments.native.Get()),
            i ? consumed_offset : written_offset, observed[i][index], observed[i][index + 1], observed[i][index + 2],
            observed[i] == expected ? "intact" : "BAD");
        return observed[i] == expected;
    };
    const auto wait = [&](UINT64 value, const char* phase) {
        check(queue->Signal(completion.Get(), value), "signal provenance fence");
        const auto deadline = GetTickCount64() + 10000;
        while (completion->GetCompletedValue() < value && GetTickCount64() < deadline) Sleep(1);
        check(device->GetDeviceRemovedReason(), "provenance device completion");
        check(completion->GetCompletedValue() >= value ? S_OK : E_FAIL, "wait provenance fence");
        std::printf("provenance fence phase=%s queue=%p fence=%p value=%llu completed=%llu\n",
            phase, static_cast<void*>(queue.Get()), static_cast<void*>(completion.Get()), value,
            completion->GetCompletedValue());
    };
    {
        ComputeCommands commands(device, queue.Get(), nullptr);
        auto* list = commands.list.Get();
        transition(list, arguments.native.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(arguments.native.Get(), 0, upload.Get(), 0, sizeof(seed));
        transition(list, input.native.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->SetComputeRootSignature(writer_root.Get());
        list->SetPipelineState(seed_pipeline.Get());
        list->SetComputeRootUnorderedAccessView(1, input.native->GetGPUVirtualAddress());
        std::printf("provenance input-write queue=%p resource=%p offset=%llu value=2 dispatch=1,1,1\n",
            static_cast<void*>(queue.Get()), static_cast<void*>(input.native.Get()), input_offset);
        list->Dispatch(1, 1, 1);
        transition(list, input.native.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(input_snapshot.Get(), 0, input.native.Get(), 0, 64);
        transition(list, input.native.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(list, arguments.native.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        ID3D12DescriptorHeap* heaps[]{native_heap.Get()};
        list->SetDescriptorHeaps(1, heaps);
        auto bound = native_heap->GetGPUDescriptorHandleForHeapStart();
        list->SetComputeRootDescriptorTable(0, bound);
        list->SetComputeRootUnorderedAccessView(1, arguments.native->GetGPUVirtualAddress());
        list->SetPipelineState(update_pipeline.Get());
        std::printf("provenance producer queue=%p resource=%p offset=%llu bytes=12 dispatch=1,1,1\n",
            static_cast<void*>(queue.Get()), static_cast<void*>(arguments.native.Get()), written_offset);
        list->Dispatch(1, 1, 1);
        transition(list, arguments.native.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        snapshots[0] = make_buffer(device, sizeof(seed), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(snapshots[0].Get(), 0, arguments.native.Get(), 0, sizeof(seed));
        transition(list, arguments.native.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        transition(list, input.native.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        check(list->Close(), "close provenance writer");
        ID3D12CommandList* lists[]{list};
        queue->ExecuteCommandLists(1, lists);
        wait(1, "post-write");
    }
    const D3D12_RANGE input_range{input_offset, input_offset + 20};
    check(input_snapshot->Map(0, &input_range, &memory), "map provenance scalar input");
    UINT launch = 0, adjacent = 0;
    std::memcpy(&launch, static_cast<const char*>(memory) + input_offset, sizeof(launch));
    std::memcpy(&adjacent, static_cast<const char*>(memory) + input_offset + 16, sizeof(adjacent));
    input_snapshot->Unmap(0, &none);
    std::printf("provenance input-record resource=%p offset=%llu value=%u adjacent_offset=%llu adjacent_value=%u\n",
        static_cast<void*>(input.native.Get()), input_offset, launch, input_offset + 16, adjacent);
    if (launch != 2 || adjacent != 0 || !read_record(0)) return false;
    // Separate consumer submission. The second copy is immediately before the
    // guest ExecuteIndirect and uses the exact resource/offset being consumed.
    const D3D12_RESOURCE_DESC target_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1,
        format, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    const D3D12_CLEAR_VALUE clear_value{format, {0, 0, 0, 1}};
    const auto target = resource(device, target_desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, &clear_value);
    ComPtr<ID3D12DescriptorHeap> rtv_heap;
    const D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    check(device->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heap)), "create provenance RTV heap");
    const auto rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    guest->CreateRenderTargetView(target.Get(), nullptr, rtv);
    ComPtr<ID3D12CommandAllocator> allocator;
    check(guest->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "create provenance allocator");
    const auto wrapped = begin_guest_list(guest, allocator.Get());
    const auto native_list = native_object<ID3D12GraphicsCommandList>(device, guest, wrapped.Get());
    auto* list = native_list.Get();
    using Bind = void (WINAPI*)(IUnknown*, IUnknown*);
    using Targets = void (WINAPI*)(IUnknown*, UINT, const D3D12_CPU_DESCRIPTOR_HANDLE*, BOOL, const D3D12_CPU_DESCRIPTOR_HANDLE*);
    using Clear = void (WINAPI*)(IUnknown*, SIZE_T, const float*, UINT, const D3D12_RECT*);
    const float clear[]{0, 0, 0, 1};
    slot<Clear>(wrapped.Get(), 52)(wrapped.Get(), rtv.ptr, clear, 0, nullptr);
    slot<Targets>(wrapped.Get(), 50)(wrapped.Get(), 1, &rtv, FALSE, nullptr);
    slot<Bind>(wrapped.Get(), 34)(wrapped.Get(), graphics_root);
    slot<Bind>(wrapped.Get(), 25)(wrapped.Get(), graphics);
    const D3D12_VIEWPORT viewport{0, 0, width, height, 0, 1};
    const D3D12_RECT scissor{0, 0, width, height};
    slot<void (WINAPI*)(IUnknown*, UINT, const D3D12_VIEWPORT*)>(wrapped.Get(), 21)(wrapped.Get(), 1, &viewport);
    slot<void (WINAPI*)(IUnknown*, UINT, const D3D12_RECT*)>(wrapped.Get(), 22)(wrapped.Get(), 1, &scissor);
    // Snapshot all guard words too; report only the consumed 12-byte interval.
    snapshots[1] = make_buffer(device, sizeof(seed), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    transition(list, arguments.native.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(snapshots[1].Get(), 0, arguments.native.Get(), 0, sizeof(seed));
    transition(list, arguments.native.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    std::printf("provenance consumer queue=%p resource=%p offset=%llu bytes=12\n",
        static_cast<void*>(queue.Get()), static_cast<void*>(arguments.native.Get()), consumed_offset);
    using Execute = void (WINAPI*)(IUnknown*, IUnknown*, UINT, IUnknown*, UINT64, IUnknown*, UINT64);
    slot<Execute>(wrapped.Get(), 61)(wrapped.Get(), signature, 1, arguments.wrapped.Get(), consumed_offset, nullptr, 0);
    check(slot<HRESULT (WINAPI*)(IUnknown*)>(wrapped.Get(), 9)(wrapped.Get()), "close provenance consumer");
    ID3D12CommandList* lists[]{list};
    queue->ExecuteCommandLists(1, lists);
    wait(2, "pre-consume-copy-and-mesh");
    if (!read_record(1) || observed[0] != observed[1]) return false;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 bytes = 0;
    device->GetCopyableFootprints(&target_desc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
    const auto pixels = make_buffer(device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    ComputeCommands capture(device, queue.Get(), nullptr);
    transition(capture.list.Get(), target.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION source{target.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    D3D12_TEXTURE_COPY_LOCATION destination{pixels.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    destination.PlacedFootprint = footprint;
    capture.list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    check(capture.list->Close(), "close provenance pixel capture");
    ID3D12CommandList* captures[]{capture.list.Get()};
    queue->ExecuteCommandLists(1, captures);
    wait(3, "scene-color-readback");
    const D3D12_RANGE pixel_range{0, static_cast<SIZE_T>(bytes)};
    check(pixels->Map(0, &pixel_range, &memory), "map provenance pixels");
    UINT changed = 0;
    bool correct = true;
    for (UINT y = 0; y < height; ++y) for (UINT x = 0; x < width; ++x) {
        UINT actual = 0;
        std::memcpy(&actual, static_cast<const char*>(memory) + footprint.Offset + y * footprint.Footprint.RowPitch + 4 * x, 4);
        const UINT wanted = consumed_x ? (x < width / 2 ? red : green) : black;
        changed += actual != black;
        if (actual != wanted) {
            std::fprintf(stderr, "provenance pixel %u,%u: %08x != %08x\n", x, y, actual, wanted);
            correct = false;
        }
    }
    pixels->Unmap(0, &none);
    std::printf("provenance result case=%s changed=%u total=%u writer_x=%u consumer_x=%u\n",
        label, changed, width * height, expected_x, consumed_x);
    ComPtr<ID3D12InfoQueue> messages;
    if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&messages)))) {
        for (UINT64 i = 0; i < messages->GetNumStoredMessagesAllowedByRetrievalFilter(); ++i) {
            SIZE_T size = 0;
            check(messages->GetMessage(i, nullptr, &size), "size provenance debug message");
            std::vector<char> storage(size);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            check(messages->GetMessage(i, message, &size), "read provenance debug message");
            if (message->Severity == D3D12_MESSAGE_SEVERITY_ERROR ||
                message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION) {
                std::fprintf(stderr, "provenance debug error: %s\n", message->pDescription);
                correct = false;
            }
        }
    }
    return correct;
}
