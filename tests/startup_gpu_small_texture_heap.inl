bool small_texture_heap_test()
{
    std::vector<ComPtr<ID3D12Resource>> textures;
    constexpr DXGI_FORMAT formats[] = {DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R32_FLOAT,
                                       DXGI_FORMAT_R8G8B8A8_UINT, DXGI_FORMAT_R8G8B8A8_UNORM};
    constexpr UINT count = 96;
    auto upload = buffer(count * 512, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto readback = buffer(count * 512, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    if (!upload || !readback) return false;
    unsigned char* data = nullptr;
    if (FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&data)))) return false;
    for (UINT i = 0; i < count; ++i) std::memset(data + i*512, i+1, 4);
    upload->Unmap(0, nullptr);

    {
        x2w_startup::SmallTextureHeap pool;
        for (UINT i = 0; i < count; ++i) {
            D3D12_RESOURCE_DESC desc{};
            desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            desc.Width = 1u << (i % 5);
            desc.Height = 1u << ((i / 5) % 4);
            desc.MipLevels = 1;
            desc.DepthOrArraySize = i % 3 == 0 ? 3 : 1;
            desc.Format = formats[i % 4];
            desc.SampleDesc.Count = 1;
            auto standard = fx.device->GetResourceAllocationInfo(0, 1, &desc);
            ComPtr<ID3D12Resource> texture;
            if (!pool.place(fx.device.Get(), desc, standard.SizeInBytes, &texture) ||
                texture->GetDesc().Alignment != D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT) return false;
            textures.push_back(std::move(texture));
        }
        std::printf("small textures: %llu allocations, %llu committed bytes, %llu reserved bytes, %llu occupied bytes\n",
                    pool.count, pool.committed_equivalent, pool.reserved, pool.occupied);
        if (pool.count != count || pool.reserved >= pool.committed_equivalent || pool.occupied > pool.reserved) return false;

        // A held mip chain from a 1024-square texture begins at 128-square for base 3.
        for (const auto format : {DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC7_UNORM}) {
            auto bc = textures[0]->GetDesc();
            bc.DepthOrArraySize = 1;
            bc.Width = bc.Height = 128; bc.Format = format; bc.MipLevels = 6; bc.Alignment = 0;
            ComPtr<ID3D12Resource> texture;
            const auto standard = fx.device->GetResourceAllocationInfo(0, 1, &bc);
            const UINT64 before = pool.occupied;
            if (!pool.place(fx.device.Get(), bc, standard.SizeInBytes, &texture)) return false;
            std::printf("128x128 fmt %u six mips: %llu committed, %llu occupied\n", format, standard.SizeInBytes, pool.occupied-before);
        }
        auto denied = textures[1]->GetDesc();
        denied.Width = denied.Height = 1024;
        denied.Alignment = 0;
        ComPtr<ID3D12Resource> texture;
        auto standard = fx.device->GetResourceAllocationInfo(0, 1, &denied);
        if (pool.place(fx.device.Get(), denied, standard.SizeInBytes, &texture) || texture) return false;
        denied.Width = denied.Height = 1;
        if (pool.place(fx.device.Get(), denied, D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT, &texture) || texture) return false;
        denied.Width = denied.Height = 4;
        denied.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        if (pool.place(fx.device.Get(), denied, standard.SizeInBytes, &texture) || texture) return false;
    }

    // The allocator is gone. Write every resource before reading any so duplicate placements corrupt the check.
    begin();
    for (const bool read : {false, true}) {
        for (UINT i = 0; i < count; ++i) {
            auto* texture = textures[i].Get();
            D3D12_TEXTURE_COPY_LOCATION resource{}, footprint{};
            resource.pResource = texture;
            resource.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            footprint.pResource = read ? readback.Get() : upload.Get();
            footprint.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            footprint.PlacedFootprint = {UINT64(i) * 512, {texture->GetDesc().Format, 1, 1, 1, 256}};
            if (read) {
                D3D12_RESOURCE_BARRIER barrier{};
                barrier.Transition = {texture, 0, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE};
                fx.list->ResourceBarrier(1, &barrier);
                const D3D12_BOX pixel{0,0,0,1,1,1};
                fx.list->CopyTextureRegion(&footprint, 0,0,0, &resource, &pixel);
            } else fx.list->CopyTextureRegion(&resource, 0,0,0, &footprint, nullptr);
        }
    }
    end_and_wait();
    if (FAILED(fx.device->GetDeviceRemovedReason()) || FAILED(readback->Map(0, nullptr, reinterpret_cast<void**>(&data)))) return false;
    bool ok = true;
    for (UINT i = 0; i < count; ++i) ok &= data[i*512] == i+1;
    readback->Unmap(0, nullptr);
    return ok;
}
