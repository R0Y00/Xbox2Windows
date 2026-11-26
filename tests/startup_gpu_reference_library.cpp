#include "startup_stubs/gpu_db.h"
#include "startup_stubs/gpu_pipeline_record.h"
#include "startup_stubs/gpu_reference_library.h"
#include <dxgi1_6.h>
#include <cstdio>
#include <thread>

using namespace x2w_gpu;
using Microsoft::WRL::ComPtr;
static int check_entry_locks()
{
    VpReferenceEntryLocks locks;
    const std::wstring held_name = L"00000000000000000000000000000000";
    std::wstring other_name;
    // Exercise two names that would have occupied the same legacy stripe.
    for (unsigned i = 1; ; ++i) {
        other_name = std::to_wstring(i);
        if (std::hash<std::wstring>{}(other_name) % 32 == std::hash<std::wstring>{}(held_name) % 32) break;
    }
    std::unique_lock<std::mutex> held(locks.for_name(held_name));
    bool serialized = false, independent = false;
    std::thread caller([&] {
        std::unique_lock<std::mutex> duplicate(locks.for_name(held_name), std::try_to_lock);
        std::unique_lock<std::mutex> unrelated(locks.for_name(other_name), std::try_to_lock);
        serialized = !duplicate.owns_lock();
        independent = unrelated.owns_lock();
    });
    caller.join();
    return serialized && independent ? 0 : 1;
}
int wmain(int argc, wchar_t** argv)
{
    if (argc == 2 && wcscmp(argv[1], L"--entry-locks") == 0) return check_entry_locks();
    if (argc != 4 || wcslen(argv[3]) != 16) return 2;
    SetEnvironmentVariableW(L"X2W_GPU_DB", argv[1]);
    std::string key(argv[3], argv[3] + 16);
    auto* record = gpu_shader_by_key(key.c_str(), GpuStage::compute);
    if (!record) return 3;
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) ||
        FAILED(factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter))) ||
        FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) return 4;
    auto view = vp_record_view(record, vp_stage_compute);
    VpRootOptions options;
    options.aniso_point_mip = vp_device_caps(device.Get()).aniso_point_mip;
    auto layout = vp_root_layout(&view, 1, true, options);
    std::vector<unsigned char> root_blob;
    std::string error;
    if (FAILED(vp_serialize_root(layout, root_blob, error))) return 5;
    ComPtr<ID3D12RootSignature> root;
    if (FAILED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root)))) return 6;
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root.Get();
    desc.CS = {view.dxil, view.dxil_size};
    std::wstring name = vp_compute_library_name(root_blob, desc.CS);
    std::printf("root=%zu dxil=%zu name=%ls\n", root_blob.size(), view.dxil_size, name.c_str());
    VpReferenceLibrary library;
    HRESULT hr = library.open(device.Get(), argv[2]);
    std::printf("CreatePipelineLibrary=0x%08lx\n", static_cast<unsigned long>(hr));
    if (FAILED(hr)) return 7;
    ComPtr<ID3D12PipelineState> pso;
    auto start = GetTickCount64();
    hr = library.load(name, desc, pso.GetAddressOf());
    std::printf("LoadComputePipeline=0x%08lx elapsed=%llu ms object=%d device=0x%08lx\n", static_cast<unsigned long>(hr),
                GetTickCount64()-start, pso.Get()!=nullptr, static_cast<unsigned long>(device->GetDeviceRemovedReason()));
    return SUCCEEDED(hr) && pso && SUCCEEDED(device->GetDeviceRemovedReason()) ? 0 : 8;
}
