// Argument conversion for an indirect Draw with the console-only rectangle-list topology (0x11).
//
// A rectangle list names three vertices per rectangle, so the title's GPU-written VertexCountPerInstance is
// three times the rectangle count. The desktop draw covers each rectangle with two triangles (six vertices),
// so the count has to become rectangles * 6 on the GPU, after the title's producer has written it. A
// one-thread compute pass reads the title's four argument words and writes the converted four words into a
// scratch buffer that the desktop ExecuteIndirect then reads. The title's own argument buffer is left as it is.
//
// The conversion mirrors the direct draw (`count / 3 * 6`, a trailing partial rectangle is dropped). A count
// whose converted value would not fit 32 bits becomes 0 vertices (nothing is drawn) instead of wrapping.
// Python model: tools/rectlist_topology.py convert_indirect_draw_arguments.
#pragma once

#include <d3d12.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <memory>
#include <mutex>
#include <vector>

namespace x2w_startup {

inline constexpr char rect_indirect_args_hlsl[] = R"(
ByteAddressBuffer source : register(t0);
RWByteAddressBuffer converted : register(u0);

[numthreads(1, 1, 1)]
void convert() {
    uint4 args = source.Load4(0);
    uint rectangles = args.x / 3;
    // 0x2AAAAAAA * 6 is the largest product below 2^32.
    uint vertices = rectangles > 0x2AAAAAAAu ? 0 : rectangles * 6;
    converted.Store4(0, uint4(vertices, args.y, args.z, args.w));
}
)";

class RectIndirectArguments {
public:
    // Records the conversion of the four draw words at `source_address` and returns the scratch resource now
    // in INDIRECT_ARGUMENT state, or nullptr if the pass could not be set up (nothing was recorded then).
    // The scratch buffer rests in COMMON between draws; `finish` puts it back there.
    ID3D12Resource* convert(ID3D12GraphicsCommandList* list, D3D12_GPU_VIRTUAL_ADDRESS source_address)
    {
        Microsoft::WRL::ComPtr<ID3D12Device> device;
        if (!list || !source_address || FAILED(list->GetDevice(IID_PPV_ARGS(&device)))) return nullptr;
        Entry* entry = entry_for(device.Get());
        if (!entry) return nullptr;
        transition(list, entry->scratch.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->SetComputeRootSignature(entry->root.Get());
        list->SetPipelineState(entry->pipeline.Get());
        list->SetComputeRootShaderResourceView(0, source_address);
        list->SetComputeRootUnorderedAccessView(1, entry->scratch->GetGPUVirtualAddress());
        list->Dispatch(1, 1, 1);
        transition(list, entry->scratch.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        return entry->scratch.Get();
    }
    static void finish(ID3D12GraphicsCommandList* list, ID3D12Resource* scratch)
    {
        transition(list, scratch, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
    }

private:
    struct Entry {
        ID3D12Device* device = nullptr;
        Microsoft::WRL::ComPtr<ID3D12RootSignature> root;
        Microsoft::WRL::ComPtr<ID3D12PipelineState> pipeline;
        Microsoft::WRL::ComPtr<ID3D12Resource> scratch;
    };
    static void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                           D3D12_RESOURCE_STATES after)
    {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = resource;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = before;
        barrier.Transition.StateAfter = after;
        list->ResourceBarrier(1, &barrier);
    }
    // Entries are never freed: a conversion may still be queued when its list is released.
    Entry* entry_for(ID3D12Device* device)
    {
        std::lock_guard<std::mutex> guard(lock_);
        for (auto& known : entries_)
            if (known->device == device) return known->scratch ? known.get() : nullptr;
        auto entry = std::make_unique<Entry>();
        entry->device = device;
        build(device, *entry);
        entries_.push_back(std::move(entry));
        return entries_.back()->scratch ? entries_.back().get() : nullptr;
    }
    static void build(ID3D12Device* device, Entry& entry)
    {
        Microsoft::WRL::ComPtr<ID3DBlob> code, error;
        if (FAILED(D3DCompile(rect_indirect_args_hlsl, sizeof(rect_indirect_args_hlsl) - 1, "rect_indirect_args", nullptr,
                              nullptr, "convert", "cs_5_0", 0, 0, &code, &error)))
            return;
        D3D12_ROOT_PARAMETER parameters[2]{};
        parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        parameters[1].Descriptor.ShaderRegister = 0;
        for (auto& parameter : parameters) parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        const D3D12_ROOT_SIGNATURE_DESC desc{2, parameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        Microsoft::WRL::ComPtr<ID3DBlob> serialized;
        if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &serialized, &error)) ||
            FAILED(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                                               IID_PPV_ARGS(&entry.root))))
            return;
        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{};
        pipeline.pRootSignature = entry.root.Get();
        pipeline.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        if (FAILED(device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&entry.pipeline)))) return;
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = 256;
        buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        buffer.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COMMON,
                                                   nullptr, IID_PPV_ARGS(&entry.scratch))))
            entry.scratch.Reset();
    }
    std::mutex lock_;
    std::vector<std::unique_ptr<Entry>> entries_;
};

inline RectIndirectArguments& rect_indirect_arguments()
{
    static RectIndirectArguments converter;
    return converter;
}

}  // namespace x2w_startup
