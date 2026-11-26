void store_fixture(HostHeap& heap)
{
    using namespace x2w_vmemory;
    Arenas arenas;
    Recording memory;
    const UINT page = 65552;
    CHECK(packed_bytes(0x0000000381000006ull) == 256);
    CHECK(!packed_bytes(0x0000000381000007ull));
    CHECK(arenas.create(fx.device.Get()) && memory.create(fx.device.Get()));
    if (failures) return;
    Place p;
    CHECK(arenas.place(page,fx.window.Get(),window_offset,p));
    memory.base = 72;
    heap.null_raw_uav_at(memory.base);
    ComPtr<ID3D12RootSignature> root;
    CHECK(SUCCEEDED(fx.device->CreateRootSignature(0,startup_gpu_memory_store_cs,sizeof(startup_gpu_memory_store_cs),IID_PPV_ARGS(&root))));
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = root.Get();
    pd.CS = {startup_gpu_memory_store_cs,sizeof(startup_gpu_memory_store_cs)};
    ComPtr<ID3D12PipelineState> pso;
    CHECK(SUCCEEDED(fx.device->CreateComputePipelineState(&pd,IID_PPV_ARGS(&pso))));
    if (failures) return;
    auto updates = buffer(16,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_FLAG_NONE,D3D12_RESOURCE_STATE_GENERIC_READ);
    UINT* mapped = nullptr;
    CHECK(updates && SUCCEEDED(updates->Map(0,nullptr,reinterpret_cast<void**>(&mapped))));
    if (!mapped) return;
    mapped[0] = 0x81000006; mapped[1] = 3; mapped[2] = 0x12345678;
    updates->Unmap(0,nullptr);
    std::unordered_map<ID3D12Resource*,D3D12_RESOURCE_STATES> states;
    states[fx.result.Get()] = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    const auto state = [&](auto* r,D3D12_RESOURCE_STATES next) {
        auto& before = states[r];
        if (before == next) return;
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Transition = {r,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,next};
        fx.list->ResourceBarrier(1,&barrier);
        before = next;
    };
    const auto store = [&](UINT which,UINT value,UINT sample,UINT64 ordinary = 0) {
        memory.ready(fx.list.Get(),state);
        state(fx.result.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        auto* h = heap.heap();
        fx.list->SetDescriptorHeaps(1,&h);
        fx.list->SetPipelineState(pso.Get());
        fx.list->SetComputeRootSignature(root.Get());
        const UINT params[]{memory.base,which,value};
        fx.list->SetComputeRoot32BitConstants(0,3,params,0);
        fx.list->SetComputeRootShaderResourceView(1,memory.directory->GetGPUVirtualAddress());
        fx.list->SetComputeRootUnorderedAccessView(2,memory.faults->GetGPUVirtualAddress());
        fx.list->SetComputeRootUnorderedAccessView(3,fx.result->GetGPUVirtualAddress());
        fx.list->SetComputeRootShaderResourceView(4, ordinary ? ordinary : memory.zeros->GetGPUVirtualAddress());
        fx.list->Dispatch(1,1,1);
        state(fx.result.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE);
        fx.list->CopyBufferRegion(fx.readback.Get(),sample*8,fx.result.Get(),0,8);
    };
    const UINT slot = arenas.packed_slot(UINT64{page} << 16);
    const auto packed = [&](UINT frame) {
        arenas.run(fx.list.Get(),p,window_offset,slot*16,256,UINT32_MAX,frame,state);
        arenas.merge(fx.list.Get(),p,0,256,slot+2,state);
    };
    begin();
    state(fx.window.Get(),D3D12_RESOURCE_STATE_COPY_DEST);
    fx.list->CopyBufferRegion(fx.window.Get(),window_offset,updates.Get(),0,8);
    arenas.merge(fx.list.Get(),p,0,page_bytes,0,state);
    const auto view = [&](UINT i,auto* w,UINT64 size) { heap.raw_uav_at(i,w,0,size); };
    MapError error{};
    CHECK(!memory.map_to(fx.list.Get(),0,p.arena->bytes.Get(),0,true,view,state,false,&error) && error == MapError::page);
    CHECK(!memory.map_to(fx.list.Get(),page,p.arena->bytes.Get(),1,true,view,state,false,&error) && error == MapError::view);
    {
        Recording saturated;
        saturated.windows.assign(slots-1, fx.window);
        CHECK(!saturated.map_to(fx.list.Get(),page,p.arena->bytes.Get(),0,true,view,state,false,&error) &&
              error == MapError::slots);
        saturated.windows.assign(1, p.arena->bytes);
        saturated.updates = entries;
        CHECK(!saturated.map_to(fx.list.Get(),page,p.arena->bytes.Get(),0,true,view,state,false,&error) &&
              error == MapError::updates);
    }
    CHECK(memory.map_to(fx.list.Get(),page,p.arena->bytes.Get(),p.offset,true,view,state,true));
    CHECK(memory.map_to(fx.list.Get(),page+1,p.arena->bytes.Get(),p.offset,false,view,state,true));
    packed(100);
    store(page,0xfeed0123,0);
    state(fx.window.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE);
    fx.list->CopyBufferRegion(fx.readback.Get(),48,fx.window.Get(),window_offset+16,4);
    state(fx.window.Get(),D3D12_RESOURCE_STATE_COPY_DEST);
    fx.list->CopyBufferRegion(fx.window.Get(),window_offset+20,updates.Get(),8,4);
    arenas.merge(fx.list.Get(),p,0,page_bytes,0,state);
    packed(101);
    store(page+1,0xbeefaaaa,1); // read succeeds; a store through a protected mapping goes to the null UAV
    store(page,0,2);
    packed(110);                // the same packed bytes streamed again after eight frames
    store(page,0,3);
    store(page,0xabcdef01,4);
    arenas.publish(fx.list.Get(),{p,16,4},state,memory.zeros.Get());
    state(fx.window.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE);
    fx.list->CopyBufferRegion(fx.readback.Get(),56,fx.window.Get(),window_offset+12,12);
    CHECK(memory.map_to(fx.list.Get(),page+2,fx.window.Get(),window_offset,false,view,state));
    const std::vector<ID3D12Resource*> shared{fx.window.Get()};
    const auto allow = [&](const std::vector<ID3D12Resource*>& writers) {
        return memory.allow_ordinary_reads(fx.list.Get(),shared,writers,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,view,state);
    };
    CHECK(!allow(shared));
    memory.writable_mode = true;
    CHECK(!allow({}));
    memory.writable_mode = false;
    CHECK(allow({}));
    const auto ordinary = fx.window->GetGPUVirtualAddress()+window_offset+16;
    store(page+2,0x33333333,9);
    store(page+2,0x1234eeee,10,ordinary);
    memory.restore_read_aliases(fx.list.Get(),state);
    state(fx.window.Get(),D3D12_RESOURCE_STATE_COPY_DEST);
    fx.list->CopyBufferRegion(fx.window.Get(),window_offset+16,updates.Get(),8,4);
    const UINT stable = memory.updates;
    CHECK(allow({}) && memory.updates == stable);
    store(page+2,0x1234eeee,11,ordinary);
    memory.restore_read_aliases(fx.list.Get(),state);
    store(page+2,0,12);
    end_and_wait();
    CHECK(SUCCEEDED(fx.readback->Map(0,nullptr,reinterpret_cast<void**>(&mapped))));
    if (!mapped) return;
    CHECK(mapped[0]==pattern(4) && mapped[1]==pattern(5));
    CHECK(mapped[2]==0xfeed0123 && mapped[3]==0x12345678);
    CHECK(mapped[4]==0xfeed0123 && mapped[5]==0x12345678);
    CHECK(mapped[6]==pattern(4) && mapped[7]==0x12345678);
    CHECK(mapped[12]==pattern(4));
    CHECK(mapped[14]==pattern(3) && mapped[15]==0xabcdef01 && mapped[16]==0x12345678);
    CHECK(mapped[20]==0xabcdef01 && mapped[21]==0xabcdef01);
    CHECK(mapped[22]==0x12345678 && mapped[23]==0x12345678);
    CHECK(mapped[24]==0x12345678 && mapped[25]==0x12345678);
    fx.readback->Unmap(0,nullptr);
}
