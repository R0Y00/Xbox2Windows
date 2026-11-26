#pragma once
#include "startup_stubs/gpu_material_fields.h"
#include <algorithm>
#include <cstring>
#include <d3dcompiler.h>
#include <d3d12.h>
#include <array>
#include <vector>
#include <wrl/client.h>

namespace x2w_material_trace {
constexpr UINT null_index = 2097151;
inline UINT index(UINT value) { return (std::min)(value & 0x07ffffffu, null_index); }
struct Record {
    Microsoft::WRL::ComPtr<ID3D12Resource> source;
    UINT64 offset{}, bytes{};
    x2w_material::Mask fields{};
};
using Words = std::array<UINT, 16>;
template<class State>
inline void copy_record(ID3D12GraphicsCommandList* list, const Record& r, ID3D12Resource* readback,
                        D3D12_RESOURCE_STATES before, State&& state)
{
    if (!(before & D3D12_RESOURCE_STATE_COPY_SOURCE)) state(D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(readback, 0, r.source.Get(), r.offset, x2w_material::byte_count(r.fields));
    if (!(before & D3D12_RESOURCE_STATE_COPY_SOURCE)) state(before);
}
inline bool cpu_words(const Record& r, Words& words)
{
    D3D12_HEAP_PROPERTIES props{};
    D3D12_HEAP_FLAGS flags{};
    const UINT size = x2w_material::byte_count(r.fields);
    if (!size || !r.source || r.bytes < size || FAILED(r.source->GetHeapProperties(&props, &flags)) ||
        props.Type != D3D12_HEAP_TYPE_UPLOAD) return false;
    void* data{};
    D3D12_RANGE read{SIZE_T(r.offset), SIZE_T(r.offset + size)}, none{};
    if (FAILED(r.source->Map(0, &read, &data))) return false;
    std::memcpy(words.data(), static_cast<const unsigned char*>(data) + r.offset, size);
    r.source->Unmap(0, &none);
    return true;
}
struct Slot {
    UINT index{}, format{}, view_format{}, dimension{}, width{}, height{};
    UINT64 host{}, texture_address{};
    std::array<UINT, 8> words{};
    bool lazy{}, late{};
    const char* material_retry = "none";
};
struct Probe {
    Microsoft::WRL::ComPtr<ID3D12RootSignature> root;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> pipeline;
    Microsoft::WRL::ComPtr<ID3D12Resource> buffer, readback;
    UINT raw_slot{}, base{};
    x2w_material::Mask fields{};
    std::vector<Slot> slots;
};
inline bool init(ID3D12Device* device, Probe& p)
{
    using Microsoft::WRL::ComPtr;
    constexpr char source[] =
        "ByteAddressBuffer records:register(t0); RWByteAddressBuffer words:register(u0);"
        "cbuffer Params:register(b0){uint offset; uint count;};"
        "[numthreads(1,1,1)] void main(){for(uint i=0;i<count;i++)"
        "words.Store(i*4,records.Load(offset+i*4));}";
    D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable = {1, &range};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants.Num32BitValues = 2;
    D3D12_ROOT_SIGNATURE_DESC sig{3, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> root, cs;
    if (FAILED(D3D12SerializeRootSignature(&sig, D3D_ROOT_SIGNATURE_VERSION_1, &root, nullptr)) ||
        FAILED(device->CreateRootSignature(0, root->GetBufferPointer(), root->GetBufferSize(), IID_PPV_ARGS(&p.root))) ||
        FAILED(D3DCompile(source, sizeof(source)-1, "material-record", nullptr, nullptr, "main", "cs_5_0", 0, 0, &cs, nullptr)))
        return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{};
    pipeline.pRootSignature = p.root.Get();
    pipeline.CS = {cs->GetBufferPointer(), cs->GetBufferSize()};
    if (FAILED(device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&p.pipeline)))) return false;
    D3D12_HEAP_PROPERTIES heap{};
    heap.CreationNodeMask = heap.VisibleNodeMask = 1;
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = sizeof(Words);
    d.Height = d.DepthOrArraySize = d.MipLevels = d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr, IID_PPV_ARGS(&p.buffer)))) return false;
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    d.Flags = D3D12_RESOURCE_FLAG_NONE;
    return SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr, IID_PPV_ARGS(&p.readback)));
}
inline void record(ID3D12GraphicsCommandList* list, const Probe& p, D3D12_GPU_DESCRIPTOR_HANDLE raw)
{
    list->SetComputeRootSignature(p.root.Get());
    list->SetPipelineState(p.pipeline.Get());
    list->SetComputeRootDescriptorTable(0, raw);
    list->SetComputeRootUnorderedAccessView(1, p.buffer->GetGPUVirtualAddress());
    const UINT count = x2w_material::byte_count(p.fields) / 4;
    const UINT constants[]{p.base, count};
    list->SetComputeRoot32BitConstants(2, 2, constants, 0);
    list->Dispatch(1, 1, 1);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {p.buffer.Get(), 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE};
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(p.readback.Get(), 0, p.buffer.Get(), 0, count * 4);
    std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
    list->ResourceBarrier(1, &barrier);
}
}
