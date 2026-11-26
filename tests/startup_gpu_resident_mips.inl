// A protected gap in either array slice must clamp every sampled slice to the common resident
// suffix. Once the gap is committed, the original view becomes usable with the same mip numbering.
bool resident_mip_test()
{
    using namespace x2w_resident_mips;
    using Layout = x2w_startup::gfx10_local::Layout;
    auto* device = fx.device.Get();
    constexpr UINT side = 512, mips = 8, slices = 2;
    const Layout layout(x2w_startup::gfx10_local::sw_4kb_s, false, 4, side, side, slices, mips);
    if (!layout.valid) return false;
    struct Memory {
        unsigned char* p;
        ~Memory() { if (p) VirtualFree(p, 0, MEM_RELEASE); }
    } memory{static_cast<unsigned char*>(VirtualAlloc(nullptr, SIZE_T(layout.size), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE))};
    if (!memory.p) return false;
    for (UINT s = 0; s < slices; ++s)
        for (UINT mip = 0; mip < mips; ++mip) {
            const float color = float(10 + s * 100 + mip);
            for (UINT y = 0; y < side >> mip; ++y)
                for (UINT x = 0; x < side >> mip; ++x)
                    std::memcpy(memory.p + layout.address(mip, x, y, s), &color, sizeof(color));
        }
    DWORD old = 0;
    for (UINT s = 0; s < slices; ++s) {
        const auto prefix = layout.level[1 + s].offset;
        if (!VirtualProtect(memory.p + s * layout.slice_size + prefix, SIZE_T(layout.slice_size-prefix), PAGE_NOACCESS, &old))
            return false;
    }
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = d.Height = side;
    d.DepthOrArraySize = slices; d.MipLevels = mips; d.SampleDesc.Count = 1; d.Format = DXGI_FORMAT_R32_FLOAT;
    const D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
    ComPtr<ID3D12Resource> texture;
    if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&texture)))) return false;
    x2w_startup::SliceFootprints plan(device, d, mips);
    auto output = buffer(16, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    auto result = buffer(16, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    ComPtr<ID3D12DescriptorHeap> heap;
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,1,D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    if (!output || !result || FAILED(device->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap)))) return false;
    D3D12_ROOT_PARAMETER params[2]{};
    D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable = {1,&range};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    D3D12_STATIC_SAMPLER_DESC sampler{};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    D3D12_ROOT_SIGNATURE_DESC root_desc{2,params,1,&sampler};
    const char shader[] = "Texture2DArray<float> t:register(t0); SamplerState s:register(s0); RWByteAddressBuffer o:register(u0);"
        "[numthreads(1,1,1)] void main(){uint w,h,a,n;t.GetDimensions(0,w,h,a,n);float3 uv=float3(.5,.5,0);"
        "o.Store4(0,uint4(t.SampleLevel(s,uv,0),t.SampleLevel(s,uv,3),t.SampleLevel(s,uv,10),w));}";
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3DBlob> serialized, cs;
    if (FAILED(D3D12SerializeRootSignature(&root_desc,D3D_ROOT_SIGNATURE_VERSION_1,&serialized,nullptr)) ||
        FAILED(device->CreateRootSignature(0,serialized->GetBufferPointer(),serialized->GetBufferSize(),IID_PPV_ARGS(&root))) ||
        FAILED(D3DCompile(shader,sizeof(shader)-1,nullptr,nullptr,nullptr,"main","cs_5_0",0,0,&cs,nullptr))) return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC p{};
    p.pRootSignature=root.Get(); p.CS={cs->GetBufferPointer(),cs->GetBufferSize()};
    ComPtr<ID3D12PipelineState> pso;
    if (FAILED(device->CreateComputePipelineState(&p,IID_PPV_ARGS(&pso)))) return false;
    D3D12_SHADER_RESOURCE_VIEW_DESC requested{};
    requested.Format = d.Format; requested.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
    requested.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    requested.Texture2DArray.MipLevels = 5; requested.Texture2DArray.FirstArraySlice = 1; requested.Texture2DArray.ArraySize = 1;
    for (UINT promotion = 0; promotion < 2; ++promotion) {
        if (promotion && !VirtualProtect(memory.p,SIZE_T(layout.size),PAGE_READWRITE,&old)) return false;
        const auto suffix = x2w_resident_mips::find(layout,reinterpret_cast<std::uintptr_t>(memory.p),[](auto at,auto bytes) {
            for (auto end=at+bytes; at<end;) {
                MEMORY_BASIC_INFORMATION region{};
                if (!VirtualQuery(reinterpret_cast<void*>(at),&region,sizeof(region)) || region.State!=MEM_COMMIT ||
                    region.Protect!=PAGE_READWRITE) return false;
                at=reinterpret_cast<std::uintptr_t>(region.BaseAddress)+region.RegionSize;
            }
            return true;
        });
        const UINT expected = promotion ? 0 : 3;
        auto selected = requested;
        if (suffix.first != expected || !limit_view(selected,mips,suffix.first)) return false;
        auto narrow = requested;
        narrow.Texture2DArray.MipLevels=2;
        if (!promotion && limit_view(narrow,mips,expected)) return false;
        auto rebased=requested;
        rebased.Texture2DArray.MostDetailedMip=1;
        if (!promotion && limit_view(rebased,mips,expected)) return false;
        auto covered=requested;
        covered.Texture2DArray.MostDetailedMip=3;
        if (!limit_view(covered,mips,expected) || covered.Texture2DArray.ResourceMinLODClamp!=0) return false;
        device->CreateShaderResourceView(texture.Get(), &selected, heap->GetCPUDescriptorHandleForHeapStart());
        UINT64 bytes=0;
        for (UINT mip=expected; mip<mips; ++mip) bytes += (plan.bytes(mip)+511)&~511ull;
        auto upload=buffer(bytes*slices,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_FLAG_NONE,D3D12_RESOURCE_STATE_GENERIC_READ);
        void* mapped=nullptr;
        if (!upload || FAILED(upload->Map(0,nullptr,&mapped))) return false;
        begin();
        x2w_gpu::transition_all(fx.list.Get(),texture.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);
        UINT64 offset=0;
        for (UINT s=0; s<slices; ++s) {
            for (UINT mip=expected; mip<mips; ++mip) {
                if (!stage_rows(layout,suffix,mip,memory.p+s*layout.slice_size,static_cast<unsigned char*>(mapped)+offset,
                                UINT(plan.row_bytes[mip]/4),plan.rows[mip],plan.placed[mip].Footprint.RowPitch)) return false;
                plan.upload(fx.list.Get(),texture.Get(),upload.Get(),mips,slices,s,mip,offset);
                offset+=(plan.bytes(mip)+511)&~511ull;
            }
        }
        upload->Unmap(0,nullptr);
        x2w_gpu::transition_all(fx.list.Get(),texture.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        x2w_gpu::transition_all(fx.list.Get(),output.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        ID3D12DescriptorHeap* heaps[]={heap.Get()};
        fx.list->SetDescriptorHeaps(1,heaps);
        fx.list->SetComputeRootSignature(root.Get()); fx.list->SetPipelineState(pso.Get());
        fx.list->SetComputeRootDescriptorTable(0,heap->GetGPUDescriptorHandleForHeapStart());
        fx.list->SetComputeRootUnorderedAccessView(1,output->GetGPUVirtualAddress());
        fx.list->Dispatch(1,1,1);
        x2w_gpu::transition_all(fx.list.Get(),output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
        fx.list->CopyBufferRegion(result.Get(),0,output.Get(),0,16);
        x2w_gpu::transition_all(fx.list.Get(),output.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON);
        x2w_gpu::transition_all(fx.list.Get(),texture.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COMMON);
        end_and_wait();
        if (FAILED(result->Map(0,nullptr,&mapped))) return false;
        const auto* vals=static_cast<const UINT*>(mapped);
        const bool ok=vals[0]==110+expected && vals[1]==113 && vals[2]==114 && vals[3]==side;
        if (!ok) std::printf("mip %u sampled %u %u %u width %u\n",expected,vals[0],vals[1],vals[2],vals[3]);
        const D3D12_RANGE none{}; result->Unmap(0,&none);
        if (!ok) return false;
    }
    // Sample an actual smaller texture, then the full replacement, through the same title view.
    // Rebind the first native texture afterwards to check a retained descriptor's mip numbering.
    d.DepthOrArraySize = 1;
    const Layout one(layout.mode, false, 4, side, side, 1, mips);
    auto view = D3D12_SHADER_RESOURCE_VIEW_DESC{};
    view.Format = d.Format;
    view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    view.Texture2D.MostDetailedMip = 3;
    view.Texture2D.MipLevels = 4;
    const UINT base = host_base(d, view, one, false);
    if (base != 3 || host_base(d, requested, one, false) != 0) return false;
    const char sampled[] =
        "Texture2D<float> t:register(t0); SamplerState s:register(s0); RWByteAddressBuffer o:register(u0);"
        "[numthreads(1,1,1)] void main(){uint w,h,n;t.GetDimensions(0,w,h,n);"
        "o.Store4(0,uint4(w,t.Load(int3(0,0,1)),t.SampleLevel(s,float2(.5,.5),0),n));}";
    cs.Reset();
    if (FAILED(D3DCompile(sampled, sizeof(sampled)-1, nullptr, nullptr, nullptr, "main", "cs_5_0", 0, 0, &cs, nullptr))) return false;
    p.CS = {cs->GetBufferPointer(), cs->GetBufferSize()};
    pso.Reset();
    if (FAILED(device->CreateComputePipelineState(&p, IID_PPV_ARGS(&pso)))) return false;
    ComPtr<ID3D12Resource> held[2];
    const auto full_bytes = device->GetResourceAllocationInfo(0, 1, &d).SizeInBytes;
    for (UINT pass = 0; pass < 3; ++pass) {
        const UINT selected_base = pass == 1 ? 0 : base;
        const auto native = host_desc(d, selected_base);
        auto& host = held[pass == 1];
        auto selected = view;
        rebase(selected, selected_base);
        if (pass != 1 && device->GetResourceAllocationInfo(0, 1, &native).SizeInBytes >= full_bytes) return false;
        if (!host && FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &native, D3D12_RESOURCE_STATE_COMMON,
                                                             nullptr, IID_PPV_ARGS(&host)))) return false;
        device->CreateShaderResourceView(host.Get(), &selected, heap->GetCPUDescriptorHandleForHeapStart());
        const x2w_startup::SliceFootprints native_plan(device, native, native.MipLevels);
        const Suffix cut{selected_base, prefix_bytes(one, selected_base)};
        UINT64 bytes = 0;
        for (UINT i = 0; i < native.MipLevels; ++i) bytes += (native_plan.bytes(i) + 511) & ~511ull;
        auto upload = buffer(bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* mapped = nullptr;
        if (!upload || FAILED(upload->Map(0, nullptr, &mapped))) return false;
        begin();
        if (pass < 2) {
            x2w_gpu::transition_all(fx.list.Get(), host.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            UINT64 offset = 0;
            for (UINT i = 0; i < native.MipLevels; ++i) {
                if (!stage_rows(one, cut, selected_base + i, memory.p, static_cast<unsigned char*>(mapped) + offset,
                                UINT(native_plan.row_bytes[i] / 4), native_plan.rows[i], native_plan.placed[i].Footprint.RowPitch)) return false;
                native_plan.upload(fx.list.Get(), host.Get(), upload.Get(), native.MipLevels, 1, 0, i, offset);
                offset += (native_plan.bytes(i) + 511) & ~511ull;
            }
            x2w_gpu::transition_all(fx.list.Get(), host.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        upload->Unmap(0, nullptr);
        ID3D12DescriptorHeap* heaps[] = {heap.Get()};
        fx.list->SetDescriptorHeaps(1, heaps);
        fx.list->SetComputeRootSignature(root.Get());
        fx.list->SetPipelineState(pso.Get());
        fx.list->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
        fx.list->SetComputeRootUnorderedAccessView(1, output->GetGPUVirtualAddress());
        x2w_gpu::transition_all(fx.list.Get(), output.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        fx.list->Dispatch(1, 1, 1);
        x2w_gpu::transition_all(fx.list.Get(), output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        fx.list->CopyBufferRegion(result.Get(), 0, output.Get(), 0, 16);
        x2w_gpu::transition_all(fx.list.Get(), output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        end_and_wait();
        if (FAILED(result->Map(0, nullptr, &mapped))) return false;
        const auto* vals = static_cast<const UINT*>(mapped);
        const bool ok = vals[0] == 64 && vals[1] == 14 && vals[2] == 13 && vals[3] == 4;
        std::printf("held mip native_base=%u native_bytes=%llu full_bytes=%llu width=%u load=%u sample=%u levels=%u\n",
                    selected_base, device->GetResourceAllocationInfo(0, 1, &native).SizeInBytes, full_bytes, vals[0], vals[1], vals[2], vals[3]);
        const D3D12_RANGE none{};
        result->Unmap(0, &none);
        if (!ok) return false;
    }
    return true;
}
