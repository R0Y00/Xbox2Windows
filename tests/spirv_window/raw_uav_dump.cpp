// Dispatch a compute shader with the sampler fixture's bindings and write its whole raw UAV to a file.
//
// The bindings match tests/spirv_window/sampler_readback.cpp: t0 is a 2x1 R32_FLOAT texture holding
// 0.25 and 0.75, u0 is a raw buffer, s1-s2 (space0) are point samplers with wrap and clamp U addressing,
// and s1-s2 (space3) are LESS_EQUAL comparison samplers. Before the dispatch every word i of the UAV
// holds raw_uav_fill_word(i), so words the shader does not write remain distinguishable from words
// it writes. The caller compares the file; this function checks nothing about the values.
#include "../desktop_gpu.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

// Keep in step with FILL_MULTIPLIER in tests/dxc_gvn_output_check.py.
uint32_t raw_uav_fill_word(uint32_t index)
{
    return (index + 1u) * 0x9E3779B1u | 1u;
}

}  // namespace

void dump_raw_uav(ID3D12Device* device, const char* shader_path, const char* groups_text,
                  const char* bytes_text, const char* out_path)
{
    std::ifstream stream(shader_path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open raw UAV shader");
    std::vector<char> shader(std::istreambuf_iterator<char>(stream), {});
    const unsigned long groups = std::stoul(groups_text);
    const unsigned long bytes = std::stoul(bytes_text);
    if (groups < 1 || groups > 2) throw std::runtime_error("Raw UAV dump takes 1 or 2 groups");
    if (bytes < 16 || bytes % 16 != 0 || bytes > (1u << 24)) throw std::runtime_error("Raw UAV size");

    D3D12_DESCRIPTOR_RANGE ranges[] = {
        {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0},
        {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0},
        {D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 2, 1, 0, 0},
        {D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 2, 1, 3, 0},
    };
    D3D12_ROOT_PARAMETER params[4]{};
    for (UINT i = 0; i < 4; ++i) {
        params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[i].DescriptorTable = {1, &ranges[i]};
    }
    D3D12_ROOT_SIGNATURE_DESC desc{4, params};
    ComPtr<ID3DBlob> serialized, error;
    check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &error),
          "Serialize raw UAV root");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                                      IID_PPV_ARGS(&root)), "Raw UAV root");
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = root.Get();
    pd.CS = {shader.data(), shader.size()};
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pipeline)), "Raw UAV pipeline");

    D3D12_RESOURCE_DESC td{};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = 2;
    td.Height = td.DepthOrArraySize = td.MipLevels = td.SampleDesc.Count = 1;
    td.Format = DXGI_FORMAT_R32_FLOAT;
    auto texture = resource(device, td, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 texture_upload_size{};
    device->GetCopyableFootprints(&td, 0, 1, 0, &footprint, nullptr, nullptr, &texture_upload_size);
    const UINT64 fill_offset = (texture_upload_size + 255) & ~UINT64(255);
    auto upload = make_buffer(device, fill_offset + bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped = nullptr;
    D3D12_RANGE empty{};
    check(upload->Map(0, &empty, &mapped), "Map raw UAV upload");
    const float texels[] = {0.25f, 0.75f};
    std::memcpy(mapped, texels, sizeof(texels));
    auto* fill = reinterpret_cast<uint32_t*>(static_cast<char*>(mapped) + fill_offset);
    for (uint32_t i = 0; i < bytes / 4; ++i) fill[i] = raw_uav_fill_word(i);
    upload->Unmap(0, nullptr);

    // Created in COMMON; the fill copy promotes it to COPY_DEST implicitly.
    auto output = make_buffer(device, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto readback = make_buffer(device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    ComPtr<ID3D12DescriptorHeap> views, samplers;
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&views)), "Create raw UAV views");
    auto cpu = views->GetCPUDescriptorHandleForHeapStart();
    device->CreateShaderResourceView(texture.Get(), nullptr, cpu);
    const UINT view_stride = device->GetDescriptorHandleIncrementSize(hd.Type);
    cpu.ptr += view_stride;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Format = DXGI_FORMAT_R32_TYPELESS;
    uav.Buffer.NumElements = static_cast<UINT>(bytes / 4);
    uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    device->CreateUnorderedAccessView(output.Get(), nullptr, &uav, cpu);
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    hd.NumDescriptors = 4;
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&samplers)), "Create raw UAV samplers");
    const UINT sampler_stride = device->GetDescriptorHandleIncrementSize(hd.Type);
    auto handle = samplers->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < 4; ++i) {
        const bool comparison = i >= 2;
        D3D12_SAMPLER_DESC sampler{};
        sampler.Filter = comparison ? D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT : D3D12_FILTER_MIN_MAG_MIP_POINT;
        sampler.AddressU = (i & 1) ? D3D12_TEXTURE_ADDRESS_MODE_CLAMP : D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.ComparisonFunc = comparison ? D3D12_COMPARISON_FUNC_LESS_EQUAL : D3D12_COMPARISON_FUNC_NEVER;
        sampler.MaxAnisotropy = 1;
        device->CreateSampler(&sampler, handle);
        handle.ptr += sampler_stride;
    }

    ComputeCommands commands(device, pipeline.Get());
    auto list = commands.list.Get();
    D3D12_TEXTURE_COPY_LOCATION dst{texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    D3D12_TEXTURE_COPY_LOCATION src{upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    src.PlacedFootprint = footprint;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    list->CopyBufferRegion(output.Get(), 0, upload.Get(), fill_offset, bytes);
    transition(list, texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(list, output.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->SetComputeRootSignature(root.Get());
    ID3D12DescriptorHeap* heaps[] = {views.Get(), samplers.Get()};
    list->SetDescriptorHeaps(2, heaps);
    auto gpu = views->GetGPUDescriptorHandleForHeapStart();
    list->SetComputeRootDescriptorTable(0, gpu);
    gpu.ptr += view_stride;
    list->SetComputeRootDescriptorTable(1, gpu);
    gpu = samplers->GetGPUDescriptorHandleForHeapStart();
    list->SetComputeRootDescriptorTable(2, gpu);
    gpu.ptr += 2 * sampler_stride;
    list->SetComputeRootDescriptorTable(3, gpu);
    list->Dispatch(static_cast<UINT>(groups), 1, 1);
    transition(list, output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(readback.Get(), 0, output.Get(), 0, bytes);
    commands.submit_and_wait(device);

    D3D12_RANGE read{0, static_cast<SIZE_T>(bytes)};
    check(readback->Map(0, &read, &mapped), "Map raw UAV readback");
    std::ofstream out(out_path, std::ios::binary);
    out.write(static_cast<const char*>(mapped), static_cast<std::streamsize>(bytes));
    readback->Unmap(0, &empty);
    if (!out) throw std::runtime_error("Cannot write raw UAV dump");
    std::printf("raw uav dump: groups=%lu bytes=%lu out=%s\n", groups, bytes, out_path);
}
