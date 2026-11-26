#pragma once

#include "startup_stubs/gpu_ambient_kernels.h"
#include <d3d12.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <string>

namespace x2w_gpu::ambient {

enum class Role { none, resolve, radiance, air };
struct Settings {
    float level = 0, constant = 0, strength = 1, radius = 0.6f, air = 0;
};
struct Views {
    D3D12_GPU_DESCRIPTOR_HANDLE depth{}, image{};
    UINT width = 0, height = 0, slices = 1;
};

// Own one instance per device. The caller serializes recording, binds the shader-visible
// heap and prepares the image for unordered access and the resolve depth for compute reads.
class Pass {
public:
    HRESULT error() const { return status_; }
    const std::string& diagnostic() const { return diagnostic_; }

    bool record(ID3D12Device* device, ID3D12GraphicsCommandList* list, Role role,
                const Settings& settings, const Views& views)
    {
        if (!device || !list || role == Role::none || (!views.width || !views.height || !views.image.ptr) ||
            !(settings.level > 0 || settings.constant > 0)) return false;
        if (role == Role::resolve && !views.depth.ptr) return false;
        if (role != Role::radiance && settings.constant == 0 && !measured_) return false;
        if (role == Role::air && (!views.slices || settings.constant > 0 || !(settings.air > 0))) return false;
        if (status_ == S_FALSE) status_ = initialize(device);
        if (FAILED(status_)) return false;

        const struct Parameters {
            float intensity, fixed, strength, radius, near_plane, vertical_slope;
            UINT width, height;
            float adaptation, maximum_ratio, fog_ratio, reserved;
        } data{settings.level, settings.constant, settings.strength, settings.radius, 0.1f, 0.45f,
               views.width, views.height, 0.04f, 4.0f, settings.air, 0.0f};
        static_assert(sizeof(data) == 12 * sizeof(UINT));

        D3D12_RESOURCE_BARRIER order{};
        order.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        list->ResourceBarrier(1, &order);
        list->SetComputeRootSignature(root_.Get());
        list->SetComputeRoot32BitConstants(0, 12, &data, 0);
        if (role == Role::resolve) list->SetComputeRootDescriptorTable(1, views.depth);
        list->SetComputeRootDescriptorTable(2, views.image);
        list->SetComputeRootUnorderedAccessView(3, history_->GetGPUVirtualAddress());
        list->SetComputeRootUnorderedAccessView(4, probes_->GetGPUVirtualAddress());
        const auto dispatch = [&](Kernel kernel, UINT x, UINT y, UINT z) {
            list->SetPipelineState(pipelines_[kernel].Get());
            list->Dispatch(x, y, z);
            list->ResourceBarrier(1, &order);
        };
        const auto groups = [](UINT size, UINT width) { return (size - 1) / width + 1; };
        switch (role) {
        case Role::radiance:
            dispatch(measure, 8, 9, 1);
            dispatch(adapt, 1, 1, 1);
            measured_ = true;
            break;
        case Role::resolve:
            if (settings.constant == 0) {
                dispatch(locate, 8, 9, 1);
                dispatch(integrate, 20, 45, 1);
            }
            dispatch(resolve, groups(views.width, 8), groups(views.height, 8), 1);
            break;
        case Role::air:
            dispatch(limit_fog, groups(views.width, 8), groups(views.height, 8), groups(views.slices, 2));
            break;
        case Role::none: return false;
        }
        return true;
    }

private:
    template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
    enum Kernel { measure, adapt, locate, integrate, resolve, limit_fog, count };
    HRESULT initialize(ID3D12Device* device)
    {
        D3D12_DESCRIPTOR_RANGE srv{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0};
        D3D12_DESCRIPTOR_RANGE uav{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0};
        D3D12_ROOT_PARAMETER params[5]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants.Num32BitValues = 12;
        params[1].ParameterType = params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[1].DescriptorTable = {1, &srv};
        params[2].DescriptorTable = {1, &uav};
        params[3].ParameterType = params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[3].Descriptor.ShaderRegister = 1;
        params[4].Descriptor.ShaderRegister = 2;
        D3D12_ROOT_SIGNATURE_DESC signature{5, params};
        ComPtr<ID3DBlob> bytes, messages;
        HRESULT hr = D3D12SerializeRootSignature(&signature, D3D_ROOT_SIGNATURE_VERSION_1, &bytes, &messages);
        if (FAILED(hr)) return hr;
        hr = device->CreateRootSignature(0, bytes->GetBufferPointer(), bytes->GetBufferSize(), IID_PPV_ARGS(&root_));
        if (FAILED(hr)) return hr;

        const struct Program { const char* part; const char* entry; } programs[count] = {
            {"MEASURE", "Measure"}, {"MEASURE", "Adapt"}, {"PROBES", "Locate"}, {"PROBES", "Integrate"},
            {"RESOLVE", "ResolveDiffuse"}, {"SCATTER", "LimitFog"}
        };
        for (UINT i = 0; i < count; ++i) {
            const D3D_SHADER_MACRO defines[] = {{programs[i].part, "1"}, {}};
            bytes.Reset();
            messages.Reset();
            hr = D3DCompile(detail::source, sizeof(detail::source)-1, "gpu_ambient", defines, nullptr,
                            programs[i].entry, "cs_5_1", 0, 0, &bytes, &messages);
            if (FAILED(hr)) {
                diagnostic_ = programs[i].entry;
                if (messages) diagnostic_.append(": ").append(static_cast<const char*>(messages->GetBufferPointer()),
                                                              messages->GetBufferSize());
                return hr;
            }
            D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
            pso.pRootSignature = root_.Get();
            pso.CS = {bytes->GetBufferPointer(), bytes->GetBufferSize()};
            hr = device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&pipelines_[i]));
            if (FAILED(hr)) { diagnostic_ = programs[i].entry; return hr; }
        }

        const auto make_buffer = [&](UINT64 size, ComPtr<ID3D12Resource>& result) {
            D3D12_RESOURCE_DESC d{};
            d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            d.Width = size;
            d.Height = d.DepthOrArraySize = d.MipLevels = d.SampleDesc.Count = 1;
            d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
            return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&result));
        };
        hr = make_buffer((2 * 64 * 36 + 1) * 16, history_);
        return SUCCEEDED(hr) ? make_buffer(160 * 90 * 3 * 16, probes_) : hr;
    }

    HRESULT status_ = S_FALSE;
    std::string diagnostic_;
    bool measured_ = false;
    ComPtr<ID3D12RootSignature> root_;
    std::array<ComPtr<ID3D12PipelineState>, count> pipelines_;
    ComPtr<ID3D12Resource> history_, probes_;
};
}
