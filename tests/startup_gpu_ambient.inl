// Uniform colored radiance has a known diffuse estimate. Seed conflicting probe pixels and a bright
// fog volume so the fixture distinguishes measured light, the resolve ceiling, and the air clamp.
bool ambient_lighting_test()
{
    using namespace x2w_gpu::ambient;
    using namespace DirectX;
    using namespace DirectX::PackedVector;
    auto* device = fx.device.Get();
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    ComPtr<ID3D12DescriptorHeap> heap;
    if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)))) return false;
    const UINT inc = device->GetDescriptorHandleIncrementSize(hd.Type);
    const auto cpu = [&](UINT i){ return D3D12_CPU_DESCRIPTOR_HANDLE{heap->GetCPUDescriptorHandleForHeapStart().ptr + SIZE_T(i)*inc}; };
    const auto gpu = [&](UINT i){ return D3D12_GPU_DESCRIPTOR_HANDLE{heap->GetGPUDescriptorHandleForHeapStart().ptr + UINT64(i)*inc}; };
    struct Texture {
        ComPtr<ID3D12Resource> gpu, upload, readback;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT foot{};
        UINT64 bytes{};
    };
    std::array<Texture,4> textures;
    const auto location = [](ID3D12Resource* r, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layout) {
        D3D12_TEXTURE_COPY_LOCATION loc{};
        loc.pResource = r;
        if (layout) { loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; loc.PlacedFootprint = *layout; }
        else loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        return loc;
    };
    begin();
    const auto uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    const auto read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    for (UINT i=0; i<textures.size(); ++i) {
        auto& t = textures[i];
        D3D12_RESOURCE_DESC d{};
        d.Dimension = i == 3 ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = 13; d.Height = 9; d.DepthOrArraySize = i == 3 ? 3 : 1;
        d.MipLevels = d.SampleDesc.Count = 1;
        d.Format = i == 0 ? DXGI_FORMAT_R32_FLOAT : i == 2 ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R11G11B10_FLOAT;
        d.Flags = i == 0 ? D3D12_RESOURCE_FLAG_NONE : D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
        if (FAILED(device->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_COPY_DEST,
                                                    nullptr,IID_PPV_ARGS(&t.gpu)))) return false;
        if (!i) device->CreateShaderResourceView(t.gpu.Get(),nullptr,cpu(i));
        else device->CreateUnorderedAccessView(t.gpu.Get(),nullptr,nullptr,cpu(i));
        device->GetCopyableFootprints(&d,0,1,0,&t.foot,nullptr,nullptr,&t.bytes);
        t.upload = buffer(t.bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_FLAG_NONE,D3D12_RESOURCE_STATE_GENERIC_READ);
        t.readback = buffer(t.bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_FLAG_NONE,D3D12_RESOURCE_STATE_COPY_DEST);
        void* mapped = nullptr;
        if (!t.upload || !t.readback || FAILED(t.upload->Map(0,nullptr,&mapped))) return false;
        for (UINT z=0; z<d.DepthOrArraySize; ++z) for (UINT y=0; y<d.Height; ++y) {
            auto* row = reinterpret_cast<UINT*>(static_cast<BYTE*>(mapped)+(z*d.Height+y)*t.foot.Footprint.RowPitch);
            for (UINT x=0; x<d.Width; ++x) {
                if (i == 0) row[x] = 0x3f800000;
                else if (i == 1) row[x] = XMFLOAT3PK(.25f,.5f,1.f).v;
                else if (i == 2) row[x] = x == 0 ? XMFLOAT3SE(64.f,0.f,64.f).v : x == 2 ? XMFLOAT3SE(30.f,30.f,30.f).v : 0;
                else row[x] = XMFLOAT3PK(8.f,8.f,8.f).v;
            }
        }
        t.upload->Unmap(0,nullptr);
        const auto dst=location(t.gpu.Get(),nullptr), src=location(t.upload.Get(),&t.foot);
        fx.list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
        x2w_gpu::transition_all(fx.list.Get(),t.gpu.Get(),D3D12_RESOURCE_STATE_COPY_DEST,i ? uav : read);
    }
    Pass pass;
    Settings settings{};
    settings.level=.8f; settings.air=1; settings.strength=0;
    ID3D12DescriptorHeap* heaps[] = {heap.Get()};
    fx.list->SetDescriptorHeaps(1,heaps);
    const auto run=[&](Role r, UINT slot){
        const Views v{gpu(0),gpu(slot),13,9,slot==3 ? 3u : 1u};
        const auto ok=pass.record(device,fx.list.Get(),r,settings,v);
        if (FAILED(pass.error())) std::printf("ambient error %08lx %s\n",pass.error(),pass.diagnostic().c_str());
        return ok;
    };
    if (run(Role::resolve,2) || !run(Role::radiance,1) || !run(Role::resolve,2) || !run(Role::air,3)) return false;
    const auto copy=[&](UINT i) {
        auto& t=textures[i];
        x2w_gpu::transition_all(fx.list.Get(),t.gpu.Get(),uav,D3D12_RESOURCE_STATE_COPY_SOURCE);
        const auto dst=location(t.readback.Get(),&t.foot),src=location(t.gpu.Get(),nullptr);
        fx.list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
        x2w_gpu::transition_all(fx.list.Get(),t.gpu.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,uav);
    };
    copy(2); copy(3); end_and_wait();
    const auto pixel=[&](UINT i,UINT x,UINT y,UINT z,XMFLOAT3& out){
        auto& t=textures[i]; void* mapped=nullptr;
        if (FAILED(t.readback->Map(0,nullptr,&mapped))) return false;
        const auto* row=reinterpret_cast<const UINT*>(static_cast<const BYTE*>(mapped)+(z*9+y)*t.foot.Footprint.RowPitch);
        if (i==2) { const XMFLOAT3SE rgb(row[x]); XMStoreFloat3(&out,XMLoadFloat3SE(&rgb)); }
        else { const XMFLOAT3PK rgb(row[x]); XMStoreFloat3(&out,XMLoadFloat3PK(&rgb)); }
        const D3D12_RANGE none{}; t.readback->Unmap(0,&none); return true;
    };
    const auto close_to=[](float a,float b){return std::abs(a-b)<.018f;};
    XMFLOAT3 rgb;
    const float lum=.25f*.2126f+.5f*.7152f+.0722f;
    for (UINT x : {0u,1u,12u}) {
        if (!pixel(2,x,4,0,rgb) || !close_to(rgb.x,.2f) || !close_to(rgb.y,.4f) || !close_to(rgb.z,.8f)) return false;
    }
    if (!pixel(2,2,4,0,rgb) || !close_to(rgb.x,4*.8f*lum) || !close_to(rgb.y,rgb.x) || !close_to(rgb.z,rgb.x)) return false;
    for (UINT z : {0u,2u})
        if (!pixel(3,12,8,z,rgb) || !close_to(rgb.x,lum) || !close_to(rgb.y,lum) || !close_to(rgb.z,lum)) return false;
    begin();
    fx.list->SetDescriptorHeaps(1,heaps);
    settings.constant=.3f; settings.level=0;
    if (!run(Role::resolve,2) || run(Role::air,3)) return false;
    copy(2); end_and_wait();
    return pixel(2,12,8,0,rgb) && close_to(rgb.x,.30f) && close_to(rgb.y,.294f) && close_to(rgb.z,.282f);
}
