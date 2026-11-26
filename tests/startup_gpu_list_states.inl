bool list_states_test()
{
    constexpr UINT count = 64;
    constexpr auto common = D3D12_RESOURCE_STATE_COMMON;
    constexpr auto dest = D3D12_RESOURCE_STATE_COPY_DEST;
    constexpr auto source = D3D12_RESOURCE_STATE_COPY_SOURCE;
    auto input = buffer(count * 4, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto output = buffer(count * 4, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, dest);
    if (!input || !output) return false;
    UINT* words = nullptr;
    if (FAILED(input->Map(0, nullptr, reinterpret_cast<void**>(&words)))) return false;
    for (UINT i = 0; i < count; ++i) words[i] = 5381 + i;
    input->Unmap(0, nullptr);
    std::array<ComPtr<ID3D12Resource>, count> buffers;
    for (auto& item : buffers) {
        item = buffer(4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, common);
        if (!item) return false;
    }
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = d.Height = 32;
    d.MipLevels = d.DepthOrArraySize = 2;
    d.SampleDesc.Count = 1;
    d.Format = DXGI_FORMAT_R32G8X24_TYPELESS;
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    ComPtr<ID3D12Resource> depth;
    if (FAILED(fx.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, common, nullptr,
                                                  IID_PPV_ARGS(&depth)))) return false;
    x2w_gpu::ListStates states;
    begin();
    auto* list = fx.list.Get();
    for (UINT i = 0; i < count; ++i) {
        auto* item = buffers[i].Get();
        states.require(list, item, dest, dest);
        list->CopyBufferRegion(item, 0, input.Get(), i * 4, 4);
    }
    states.require(list, depth.Get(), D3D12_RESOURCE_STATE_DEPTH_READ, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    states.require(list, depth.Get(), D3D12_RESOURCE_STATE_DEPTH_READ, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    const auto* split = states.find(depth.Get());
    if (!split || split->depth != D3D12_RESOURCE_STATE_DEPTH_READ || split->stencil != D3D12_RESOURCE_STATE_DEPTH_WRITE ||
        states.size() != count + 1) return false;
    for (const auto& item : buffers) if (states.state(item.Get()) != dest) return false;
    states.close(list);
    if (states.size() || states.find(depth.Get())) return false;
    end_and_wait();
    begin();
    for (UINT i = count; i--;) {
        auto* item = buffers[i].Get();
        states.require(list, item, source, source);
        list->CopyBufferRegion(output.Get(), i * 4, item, 0, 4);
    }
    states.require(list, depth.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    states.close(list);
    end_and_wait();
    if (FAILED(fx.device->GetDeviceRemovedReason()) || FAILED(output->Map(0, nullptr, reinterpret_cast<void**>(&words)))) return false;
    bool good = true;
    for (UINT i = 0; i < count; ++i) good &= words[i] == 5381 + i;
    output->Unmap(0, nullptr);
    return good;
}
