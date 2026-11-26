// Request the CPU-backed descriptor before a native texture exists. Then place a real
// texture at the same address and copy an opaque red BC3 block to it. The CPU bytes
// remain zero. A fresh slot-70 descriptor must sample the actual GPU-written red texel.
{
    auto* material_memory = VirtualAlloc(nullptr, 65536, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!material_memory) return 42;
    ComPtr<ID3D12Device> native;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&native)))) return 43;
    struct { D3D12_DESCRIPTOR_HEAP_DESC desc; UINT64 backing, bytes; } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2}};
    void* guest_heap{};
    if (!ok(invoke(device, 14, arg(&heap_desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&guest_heap)))) return 44;
    const auto cpu = invoke(guest_heap, 9);
    const UINT stride = native->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto material_desc = family_desc;
    material_desc.Width = material_desc.Height = 64;
    material_desc.Format = DXGI_FORMAT_BC3_UNORM;
    material_desc.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x105);
    struct { D3D12_SHADER_RESOURCE_VIEW_DESC view; UINT64 address; } input{};
    input.address = reinterpret_cast<UINT64>(material_memory);
    input.view.Format = material_desc.Format;
    input.view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    input.view.Texture2D.MipLevels = 1;
    input.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    invoke(device, 70, arg(&material_desc), arg(&input), cpu);
    void* material{};
    if (!ok(invoke(device, 133, arg(material_memory), arg(&material_desc), D3D12_RESOURCE_STATE_COMMON, 0,
                   arg(&__uuidof(ID3D12Resource)), arg(&material))) || !material) return 45;
    auto encoded_desc = family_desc;
    encoded_desc.Width = encoded_desc.Height = 1;
    encoded_desc.Format = DXGI_FORMAT_R32G32B32A32_UINT;
    auto* encoded = make_resource(device, D3D12_HEAP_TYPE_DEFAULT, encoded_desc, D3D12_RESOURCE_STATE_COMMON);
    if (!encoded || !ok(invoke(upload, 8, 0, 0, arg(&memory))) || !memory) return 46;
    const BYTE red_block[16]{255, 255, 0, 0, 0, 0, 0, 0, 0, 248, 0, 0, 0, 0, 0, 0};
    std::memcpy(memory, red_block, 16);
    invoke(upload, 9);
    if (!ok(invoke(list, 139, arg(allocator), 0))) return 47;
    auto encoded_upload = footprint(upload, 0, encoded_desc.Format);
    encoded_upload.PlacedFootprint.Footprint.Width = encoded_upload.PlacedFootprint.Footprint.Height = 1;
    const auto packed = subresource(encoded, 0), target = subresource(material, 0);
    copy(packed, encoded_upload);
    invoke(list, 85, arg(&target), 0, 0, 0, arg(&packed));
    if (!ok(invoke(list, 9)) || !submit()) return 48;
    invoke(device, 70, arg(&material_desc), arg(&input), cpu + stride);
    D3D12_DESCRIPTOR_HEAP_DESC visible_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    ComPtr<ID3D12DescriptorHeap> visible;
    check(native->CreateDescriptorHeap(&visible_desc, IID_PPV_ARGS(&visible)), "sample heap");
    // The shim may have allocated a title-side shadow heap. Copy from the
    // actual desktop heap into a shader-visible one for this native dispatch.
    constexpr GUID key{0xb5799bda, 0xe302, 0x4194, {0x87, 0x76, 0x36, 0x91, 0xf4, 0xdc, 0x44, 0x16}};
    if (!ok(invoke(device, 5, arg(&key), arg(guest_heap)))) return 50;
    ComPtr<IUnknown> object;
    UINT pointer_bytes = sizeof(IUnknown*);
    check(native->GetPrivateData(key, &pointer_bytes, object.GetAddressOf()), "desktop heap reference");
    invoke(device, 5, arg(&key), 0);
    ComPtr<ID3D12DescriptorHeap> native_heap;
    check(object.As(&native_heap), "native descriptor heap");
    auto source_handle = native_heap->GetCPUDescriptorHandleForHeapStart();
    source_handle.ptr += stride;
    native->CopyDescriptorsSimple(1, visible->GetCPUDescriptorHandleForHeapStart(), source_handle, visible_desc.Type);
    const D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1};
    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable = {1, &range};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    const D3D12_ROOT_SIGNATURE_DESC rd{2, params};
    ComPtr<ID3DBlob> serialized, blob;
    check(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, nullptr), "sample root");
    ComPtr<ID3D12RootSignature> root;
    check(native->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&root)), "root");
    constexpr char shader[] = "Texture2D<float4> t:register(t0); RWStructuredBuffer<float4> o:register(u0);"
                              "[numthreads(1,1,1)] void main(){o[0]=t.Load(int3(0,0,0));}";
    check(D3DCompile(shader, sizeof(shader)-1, nullptr, nullptr, nullptr, "main", "cs_5_0", 0, 0, &blob, nullptr), "sample shader");
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = root.Get();
    pd.CS = {blob->GetBufferPointer(), blob->GetBufferSize()};
    ComPtr<ID3D12PipelineState> pso;
    check(native->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)), "sample pso");
    auto out = make_buffer(native.Get(), 16, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto back = make_buffer(native.Get(), 16, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    ComputeCommands commands(native.Get(), pso.Get());
    ID3D12DescriptorHeap* heaps[]{visible.Get()};
    commands.list->SetDescriptorHeaps(1, heaps);
    commands.list->SetComputeRootSignature(root.Get());
    commands.list->SetComputeRootDescriptorTable(0, visible->GetGPUDescriptorHandleForHeapStart());
    commands.list->SetComputeRootUnorderedAccessView(1, out->GetGPUVirtualAddress());
    commands.list->Dispatch(1,1,1);
    transition(commands.list.Get(), out.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    commands.list->CopyBufferRegion(back.Get(), 0, out.Get(), 0, 16);
    commands.submit_and_wait(native.Get());
    void* sampled{};
    check(back->Map(0, nullptr, &sampled), "sample readback");
    const auto* rgba = static_cast<const float*>(sampled);
    const bool red = rgba[0] == 1.f && rgba[1] == 0.f && rgba[2] == 0.f && rgba[3] == 1.f;
    std::printf("slot70 after slot85 BC3 sample: %g %g %g %g\n", rgba[0], rgba[1], rgba[2], rgba[3]);
    back->Unmap(0, nullptr);
    if (!red || std::memcmp(material_memory, red_block, 16) == 0) return 49;
}
