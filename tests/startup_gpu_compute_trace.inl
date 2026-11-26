void material_feedback_test(HostHeap& heap, const u32 (&lazy)[8], ID3D12Resource* expected)
{
    constexpr UINT cpu_slot = 91, gpu_slot = 92, unused_slot = 93, source_slot = 94;
    for (UINT slot : {cpu_slot, gpu_slot, unused_slot}) {
        std::memcpy(fx.heap_memory + slot * 32, lazy, 32);
        if (slot != gpu_slot) heap.mirror(slot, false);  // a fresh title copy can precede the heap scan
    }
    auto srv = buffer_srv(DXGI_FORMAT_R32_TYPELESS, 4, 12, 0, true);
    auto upload = buffer(128, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto wrapper = buffer_wrapper(128, 128);
    fx.resources[wrapper - 1].desktop = upload.Get();
    fx.find_fails = true;
    note(source_slot, 18, wrapper, &srv, sizeof(srv));
    CHECK(heap.mirror(source_slot));
    constexpr x2w_material::Mask fields = (1u << 0) | (1u << 4) | (1u << 6);
    auto cpu_record = heap.material_record(source_slot, 8, fields);
    fx.find_fails = false;
    CHECK(cpu_record.source.Get() == upload.Get() && cpu_record.offset == 24 && cpu_record.bytes == 40);
    CHECK(!heap.material_record(source_slot, 24, fields).source && !heap.material_record(source_slot, 1, fields).source);
    CHECK(heap.material_record(source_slot, 24, 0x0004).source && !heap.material_record(source_slot, 8, 0xfeff).source);
    x2w_material_trace::Words words{cpu_slot, cpu_slot, unused_slot, ~0u, cpu_slot, ~0u, cpu_slot};
    void* ptr{};
    CHECK(SUCCEEDED(upload->Map(0, nullptr, &ptr)));
    if (!ptr) return;
    std::memcpy(static_cast<char*>(ptr) + 24, words.data(), x2w_material::byte_count(cpu_record.fields));
    upload->Unmap(0, nullptr);
    x2w_material_trace::Words read{};
    CHECK(x2w_material_trace::cpu_words(cpu_record, read) && read == words);
    CHECK(heap.materialize(nullptr, read, fields) == 1 && heap.kept_at(cpu_slot) == expected);
    CHECK(!heap.kept_at(unused_slot) && heap.materialize(nullptr, read, fields) == 0);
    wrapper = buffer_wrapper(128, 128);
    srv.Buffer.NumElements = 24;
    note(source_slot, 18, wrapper, &srv, sizeof(srv));
    CHECK(heap.mirror(source_slot));
    constexpr x2w_material::Mask wide = 0xfeff;
    auto gpu_record = heap.material_record(source_slot, 8, wide);
    CHECK(gpu_record.source.Get() == fx.window.Get() && gpu_record.offset == window_offset + 128 + 24);
    CHECK(!x2w_material_trace::cpu_words(gpu_record, read));
    if (!gpu_record.source) return;
    constexpr UINT field12 = 2111, backlog = 501;
    u32 streamed[8];
    std::copy_n(lazy, 8, streamed);
    streamed[1] += 65536;
    fixture_blocked_address = (u64{streamed[2]} << 32) | streamed[1];
    std::memcpy(fx.heap_memory + field12 * 32, streamed, 32);
    heap.mirror(field12, false);
    std::memcpy(fx.heap_memory + backlog * 32, lazy, 32);
    heap.mirror(backlog, false);
    words[3] = field12;
    words[2] = cpu_slot;
    words[8] = unused_slot;
    words[15] = gpu_slot;
    CHECK(SUCCEEDED(upload->Map(0, nullptr, &ptr)));
    std::memcpy(ptr, words.data(), sizeof(words));
    upload->Unmap(0, nullptr);
    begin();
    D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_COMMON;
    auto transition = [&](D3D12_RESOURCE_STATES next) {
        D3D12_RESOURCE_BARRIER b{};
        b.Transition = {fx.window.Get(), 0, before, next};
        fx.list->ResourceBarrier(1, &b);
        before = next;
    };
    transition(D3D12_RESOURCE_STATE_COPY_DEST);
    fx.list->CopyBufferRegion(fx.window.Get(), gpu_record.offset, upload.Get(), 0, sizeof(words));
    x2w_material_trace::copy_record(fx.list.Get(), gpu_record, fx.readback.Get(), before, transition);
    CHECK(before == D3D12_RESOURCE_STATE_COPY_DEST);
    transition(D3D12_RESOURCE_STATE_COMMON);
    end_and_wait();
    CHECK(SUCCEEDED(fx.readback->Map(0, nullptr, &ptr)));
    std::memcpy(read.data(), ptr, sizeof(read));
    fx.readback->Unmap(0, nullptr);
    CHECK(read == words && heap.materialize(nullptr, read, wide) == 1 && heap.kept_at(gpu_slot) == expected);
    CHECK(pick(heap, 7, gpu_slot, 1)[0] == 204 && !heap.kept_at(unused_slot));
    std::vector<x2w_material_trace::Slot> slots;
    heap.material_slots(slots);
    const auto pending = std::find_if(slots.begin(), slots.end(), [field12](const auto& s) { return s.index == field12; });
    CHECK(pending != slots.end() && pending->lazy && !std::strcmp(pending->material_retry, "backing_unavailable"));
    CHECK(!heap.kept_at(field12) && !heap.kept_at(backlog));
    fixture_blocked_address = 0;
    fixture_room = 0;
    fixture_needs_allocation = true;
    CHECK(!heap.materialize(nullptr, read, 1u << 3) && !fixture_creations && !heap.kept_at(field12));
    ++fixture_epoch;
    LazyBudget budget;
    heap.warm_lazy(nullptr, 100, budget, 1, 1ull << 30, 1000);
    CHECK(budget.visits == 1 && !budget.bytes && !fixture_creations && !heap.kept_at(field12));
    fixture_needs_allocation = false;
    fixture_remap_on_probe = true;
    CHECK(!heap.materialize(nullptr, read, 1u << 3) && !heap.kept_at(field12));
    partial_clamp = 0.5f;
    ++fixture_epoch;
    heap.warm_lazy(nullptr, 100, budget, 2, 1ull << 30, 1000);
    CHECK(budget.visits == 2 && budget.bytes == 4096 && !fixture_creations);
    CHECK(heap.kept_at(field12) == expected && !heap.kept_at(backlog));
    partial_clamp = 0;
    ++fixture_epoch;
    heap.warm_lazy(nullptr, 100, budget, 3, 1ull << 30, 1000);
    CHECK(budget.visits == 3 && budget.bytes == 8192 && !fixture_creations && !heap.kept_at(backlog));
    fixture_room = UINT64_MAX;
    CHECK(pick(heap, 7, field12, 1)[0] == 204);
    // The title reuses the selected slot before its retry. The replacement has
    // not appeared in a fenced record, so it returns to the regular rotation.
    streamed[1] += 65536;
    fixture_blocked_address = (u64{streamed[2]} << 32) | streamed[1];
    std::memcpy(fx.heap_memory + field12 * 32, streamed, 32);
    heap.mirror(field12, false);
    CHECK(!heap.materialize(nullptr, read, 1u << 3));
    streamed[1] += 65536;
    std::memcpy(fx.heap_memory + field12 * 32, streamed, 32);
    fixture_blocked_address = 0;
    ++fixture_epoch;
    heap.warm_lazy(nullptr, 101, budget, 1, 1ull << 30, 1000);
    CHECK(!heap.kept_at(field12));
}

bool pixel_subresource_capture_test()
{
    using Image = x2w_compute_trace::Image;
    const auto input = Image::color_box(2560,1440,{650,300});
    const auto output = Image::color_box(3840,2160,{650,300});
    const auto edge = Image::color_box(3840,2160,{1000,0});
    if (input.left != 1579 || input.top != 389 || output.left != 2368 || output.top != 584 ||
        edge.right != 3840 || edge.top != 0) return false;
    auto* device = fx.device.Get();
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = desc.Height = 16;
    desc.MipLevels = desc.DepthOrArraySize = 2;
    desc.SampleDesc.Count = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_HEAP_PROPERTIES props{D3D12_HEAP_TYPE_DEFAULT};
    ComPtr<ID3D12Resource> source;
    if (FAILED(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&source)))) return false;
    x2w_compute_trace::Image image;
    if (!image.init(device, {source.Get(), 1, 32, true, false, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM}) ||
        image.width() != 8 || image.height() != 8 || image.footprint.Footprint.Format != DXGI_FORMAT_R8G8B8A8_UNORM) return false;
    x2w_compute_trace::Image invalid;
    if (invalid.init(device, {source.Get(), 1, 32, false, false, 1, 1, DXGI_FORMAT_R32_FLOAT})) return false;
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    ComPtr<ID3D12DescriptorHeap> views;
    if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&views)))) return false;
    D3D12_RENDER_TARGET_VIEW_DESC rtv{};
    rtv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
    rtv.Texture2DArray = {1, 1, 1, 0};
    const auto handle = views->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(source.Get(), &rtv, handle);
    begin();
    for (UINT phase = 0; phase < 2; ++phase) {
        float color[]{float(phase), 1.f-float(phase), 0, 1};
        fx.list->ClearRenderTargetView(handle, color, 0, nullptr);
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Transition = {source.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE};
        fx.list->ResourceBarrier(1, &barrier);
        image.record(fx.list.Get(), phase != 0);
        std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        fx.list->ResourceBarrier(1, &barrier);
    }
    end_and_wait();
    for (UINT phase = 0; phase < 2; ++phase) {
        auto* readback = phase ? image.post.Get() : image.pre.Get();
        void* ptr{};
        if (FAILED(readback->Map(0, nullptr, &ptr))) return false;
        const auto* pixel = static_cast<const unsigned char*>(ptr);
        const bool ok = pixel[0] == 255 * phase && pixel[1] == 255 * (1 - phase);
        readback->Unmap(0, nullptr);
        if (!ok) return false;
    }
    return SUCCEEDED(device->GetDeviceRemovedReason());
}

bool material_record_test()
{
    std::istringstream rows("20 22 33 7b\n10 44 55 5\n20 66 77 f\nbad row\n");
    const auto entries = x2w_material::read_profile(rows);
    if (entries.size() != 3) return false;
    for (const auto& entry : entries) {
        unsigned char header[20]{'D','X','B','C'};
        std::memcpy(header + 4, &entry.hash_low, 8);
        std::memcpy(header + 12, &entry.hash_high, 8);
        if (x2w_material::lookup(entries, entry.key, header, sizeof(header)) != entry.fields) return false;
        header[19] ^= 0x80;
        if (x2w_material::lookup(entries, entry.key, header, sizeof(header)) != 0) return false;
    }
    if (x2w_material::byte_count(4) != 12 || x2w_material::byte_count(0xfeff) != 64) return false;
    x2w_material_trace::Probe probe;
    if (!x2w_material_trace::init(fx.device.Get(), probe)) return false;
    auto upload = buffer(64, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto gpu = buffer(64, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    if (!upload || !gpu) return false;
    const UINT words[7]{4321, 0xf8001234, 17, 0x12345678, 0x80000020, 0, 0xffffffff};
    void* ptr = nullptr;
    if (FAILED(upload->Map(0, nullptr, &ptr))) return false;
    std::memcpy(ptr, words, sizeof(words));
    upload->Unmap(0, nullptr);
    D3D12_DESCRIPTOR_HEAP_DESC heap{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    ComPtr<ID3D12DescriptorHeap> table;
    if (FAILED(fx.device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&table)))) return false;
    D3D12_SHADER_RESOURCE_VIEW_DESC raw{};
    raw.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    raw.Format = DXGI_FORMAT_R32_TYPELESS;
    raw.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    raw.Buffer.FirstElement = 4;
    raw.Buffer.NumElements = 12;
    raw.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    fx.device->CreateShaderResourceView(gpu.Get(), &raw, table->GetCPUDescriptorHandleForHeapStart());
    begin();
    probe.base = 8;
    probe.fields = 1u << 6;
    fx.list->CopyBufferRegion(gpu.Get(), 24, upload.Get(), 0, sizeof(words));
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Transition = {gpu.Get(), 0, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
    fx.list->ResourceBarrier(1, &barrier);
    ID3D12DescriptorHeap* heaps[] = {table.Get()};
    fx.list->SetDescriptorHeaps(1, heaps);
    x2w_material_trace::record(fx.list.Get(), probe, table->GetGPUDescriptorHandleForHeapStart());
    end_and_wait();
    if (FAILED(fx.device->GetDeviceRemovedReason()) || FAILED(probe.readback->Map(0, nullptr, &ptr))) return false;
    const bool correct = std::memcmp(ptr, words, sizeof(words)) == 0 &&
        x2w_material_trace::index(words[1]) == 0x1234 && x2w_material_trace::index(words[3]) == 2097151 &&
        x2w_material_trace::index(words[4]) == 32;
    probe.readback->Unmap(0, nullptr);
    return correct;
}

// Exercise the readback path on a compute queue with two different values from the same UAV.
bool compute_texture_capture_test(UINT volume_depth = 0)
{
    auto* device = fx.device.Get();
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    D3D12_COMMAND_QUEUE_DESC q{};
    q.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    if (FAILED(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue))) ||
        FAILED(device->CreateCommandAllocator(q.Type, IID_PPV_ARGS(&alloc))) ||
        FAILED(device->CreateCommandList(0, q.Type, alloc.Get(), nullptr, IID_PPV_ARGS(&list)))) return false;
    D3D12_RESOURCE_DESC tex{};
    tex.Dimension = volume_depth ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tex.Width = volume_depth ? 33 : 271;
    tex.Height = volume_depth ? 33 : 135;
    tex.DepthOrArraySize = volume_depth ? static_cast<UINT16>(volume_depth) : 1;
    tex.MipLevels = tex.SampleDesc.Count = 1;
    tex.Format = volume_depth ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R32_UINT;
    tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    ComPtr<ID3D12Resource> image;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &tex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                               nullptr, IID_PPV_ARGS(&image)))) return false;

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    ComPtr<ID3D12DescriptorHeap> views;
    if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&views)))) return false;
    device->CreateUnorderedAccessView(image.Get(), nullptr, nullptr, views->GetCPUDescriptorHandleForHeapStart());
    D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1};
    D3D12_ROOT_PARAMETER root[2]{};
    root[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root[0].DescriptorTable = {1, &range};
    root[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    root[1].Constants.Num32BitValues = 1;
    D3D12_ROOT_SIGNATURE_DESC rs{2, root};
    ComPtr<ID3DBlob> serialized, cs;
    ComPtr<ID3D12RootSignature> sig;
    if (FAILED(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, nullptr)) ||
        FAILED(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&sig))))
        return false;
    constexpr char source[] = "RWTexture2D<uint> t:register(u0); cbuffer C:register(b0){uint d;} "
        "[numthreads(1,1,1)] void main(uint3 p:SV_DispatchThreadID){t[p.xy]=d+p.x+1024*p.y;}";
    constexpr char volume_source[] = "RWTexture3D<float4> t:register(u0); cbuffer C:register(b0){uint d;} "
        "[numthreads(1,1,1)] void main(uint3 p:SV_DispatchThreadID){t[p]=float4(p,d);}";
    const char* shader_source = volume_depth ? volume_source : source;
    if (FAILED(D3DCompile(shader_source, std::strlen(shader_source), nullptr, nullptr, nullptr, "main", "cs_5_0", 0, 0, &cs, nullptr))) return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = sig.Get();
    pso_desc.CS = {cs->GetBufferPointer(), cs->GetBufferSize()};
    ComPtr<ID3D12PipelineState> pso;
    if (FAILED(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso)))) return false;

    x2w_compute_trace::Bindings bindings;
    bindings.add({image.Get(), 2, 8, true});
    bindings.add({image.Get(), 2, 8, true});
    x2w_compute_trace::Image sample;
    if (bindings.count != 1 || !sample.init(device, bindings.values[0])) return false;
    ID3D12DescriptorHeap* heaps[] = {views.Get()};
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootSignature(sig.Get());
    list->SetPipelineState(pso.Get());
    list->SetComputeRootDescriptorTable(0, views->GetGPUDescriptorHandleForHeapStart());
    for (UINT phase = 0; phase < 2; ++phase) {
        list->SetComputeRoot32BitConstant(1, volume_depth ? phase : phase * 0x200000, 0);
        list->Dispatch(UINT(tex.Width), tex.Height, tex.DepthOrArraySize);
        x2w_gpu::transition_all(list.Get(), image.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        sample.record(list.Get(), phase != 0);
        x2w_gpu::transition_all(list.Get(), image.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    ComPtr<ID3D12Fence> fence;
    if (FAILED(list->Close()) || FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) return false;
    ID3D12CommandList* batch[] = {list.Get()};
    for (UINT replay = 1; replay <= 2; ++replay) {
        queue->ExecuteCommandLists(1, batch);
        if (FAILED(queue->Signal(fence.Get(), replay)) || FAILED(fence->SetEventOnCompletion(replay, nullptr)) ||
            fence->GetCompletedValue() != replay) return false;
        for (UINT phase = 0; phase < 2; ++phase) {
            auto* readback = phase ? sample.post.Get() : sample.pre.Get();
            void* data = nullptr;
            const D3D12_RANGE read{0, SIZE_T(sample.size)};
            if (FAILED(readback->Map(0, &read, &data))) return false;
            bool ok = true;
            if (volume_depth) {
                const auto& layout = sample.footprint.Footprint;
                const UINT expected_depth = volume_depth == 33 ? 33u : 3u;
                ok = layout.Width == 33 && layout.Height == 33 && layout.Depth == expected_depth;
                for (UINT z = 0; ok && z < expected_depth; ++z) {
                    const UINT expected_z = expected_depth == 33 ? z : z * 32;
                    for (UINT y = 0; y < 33; ++y) {
                        const auto* row = reinterpret_cast<const UINT16*>(static_cast<const unsigned char*>(data) +
                            sample.footprint.Offset + SIZE_T(z * 33 + y) * layout.RowPitch);
                        for (UINT x = 0; x < 33; ++x) {
                            using DirectX::PackedVector::XMConvertHalfToFloat;
                            if (XMConvertHalfToFloat(row[x*4]) != x || XMConvertHalfToFloat(row[x*4+1]) != y ||
                                XMConvertHalfToFloat(row[x*4+2]) != expected_z || XMConvertHalfToFloat(row[x*4+3]) != phase)
                                ok = false;
                        }
                    }
                }
            } else for (UINT y = 0; y < 128; ++y) {
                const auto* row = reinterpret_cast<const UINT*>(static_cast<const unsigned char*>(data) +
                    sample.footprint.Offset + SIZE_T(y) * sample.footprint.Footprint.RowPitch);
                for (UINT x = 0; x < 256; ++x)
                    if (row[x] != phase * 0x200000 + 7 + x + (3 + y) * 1024) ok = false;
            }
            const D3D12_RANGE none{0,0};
            readback->Unmap(0, &none);
            if (!ok) return false;
        }
    }
    return true;
}

bool compute_priority_depth_capture_test()
{
    using namespace x2w_compute_trace;
    auto* device = fx.device.Get();
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = 271;
    d.Height = 135;
    d.DepthOrArraySize = d.MipLevels = d.SampleDesc.Count = 1;
    d.Format = DXGI_FORMAT_R32G8X24_TYPELESS;
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    ComPtr<ID3D12Resource> depth;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                               nullptr, IID_PPV_ARGS(&depth)))) return false;
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1};
    ComPtr<ID3D12DescriptorHeap> views;
    if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&views)))) return false;
    D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
    dsv.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    const auto handle = views->GetCPUDescriptorHandleForHeapStart();
    device->CreateDepthStencilView(depth.Get(), &dsv, handle);

    Bindings bindings;
    d.Format = DXGI_FORMAT_R32_UINT;
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    std::array<ComPtr<ID3D12Resource>, 4> colors;
    for (UINT i=0; i<colors.size(); ++i) {
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON,
                                                   nullptr, IID_PPV_ARGS(&colors[i])))) return false;
        bindings.add({colors[i].Get(), 0, i * 8, false});
        bindings.add({colors[i].Get(), 1, i * 8, true});
    }
    bindings.add({depth.Get(), 0, 464, false}, true);
    bindings.add({colors[3].Get(), 1, 176, true}, true);
    bindings.add({colors[0].Get(), 1, 32, true}, true);
    UINT prioritized = 0;
    bool picked_depth = false, picked_light = false, relabeled = false;
    for (const auto& b : bindings.values) {
        if (!b.priority) continue;
        ++prioritized;
        picked_depth |= b.texture == depth.Get() && b.offset == 464 && !b.write;
        picked_light |= b.texture == colors[3].Get() && b.offset == 176 && b.write;
        relabeled |= b.texture == colors[0].Get() && b.offset == 32 && b.write;
    }
    if (bindings.count != 6 || prioritized != 3 || !picked_depth || !picked_light || !relabeled) return false;
    Image sample, stencil;
    if (!sample.init(device, {depth.Get(),0,464,false,true,0,0,DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS}) ||
        !sample.full_depth_plane || !stencil.init(device, {depth.Get(),0,464,false,true,0,0,DXGI_FORMAT_X32_TYPELESS_G8X24_UINT,1}) ||
        !stencil.full_depth_plane || stencil.source_subresource != 1) return false;
    begin();
    fx.list->ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.875f, 23, 0, nullptr);
    const D3D12_RECT center{7,3,263,131};
    fx.list->ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.25f, 91, 1, &center);
    x2w_gpu::transition_all(fx.list.Get(), depth.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    sample.record(fx.list.Get(),false);
    stencil.record(fx.list.Get(),false);
    end_and_wait();
    void* data = nullptr;
    if (FAILED(sample.pre->Map(0,nullptr,&data))) return false;
    const auto* pixels = static_cast<const unsigned char*>(data) + sample.footprint.Offset;
    const auto stride = sample.output_row_bytes() / sample.width();
    bool ok = sample.width() == 256 && sample.height() == 128 && stride >= sizeof(float);
    for (UINT y = 0; ok && y < sample.height(); ++y) {
        const auto* row = sample.row(pixels,y,0);
        for (UINT x = 0; x < sample.width(); ++x) {
            float value;
            std::memcpy(&value, row + x * stride, sizeof(value));
            if (value != 0.25f) { ok = false; break; }
        }
    }
    const D3D12_RANGE none{};
    sample.pre->Unmap(0,&none);
    if (FAILED(stencil.pre->Map(0,nullptr,&data))) return false;
    pixels = static_cast<const unsigned char*>(data) + stencil.footprint.Offset;
    const auto bytes_per_texel = stencil.row_bytes / stencil.footprint.Footprint.Width;
    for (UINT y = 0; ok && y < stencil.height(); ++y) {
        const auto* row = stencil.row(pixels,y,0);
        for (UINT x = 0; x < stencil.width(); ++x) {
            const auto* first = row + x * bytes_per_texel;
            if (std::find(first, first + bytes_per_texel, 91u) == first + bytes_per_texel) { ok = false; break; }
        }
    }
    stencil.pre->Unmap(0,&none);
    return ok;
}
