// Direct compute dispatch and root argument bindings on a console command list.
//
//   startup_compute_dispatch <warp|hardware>     (X2W_D3D12_PERMISSIVE=1 X2W_D3D12_LIST_NOOP=1 X2W_D3D12_RENDER=1 X2W_D3D12_DEBUG=1)
//
// Verifies:
//   - CommandList slot 33: SetComputeRootSignature
//   - CommandList slot 25: SetPipelineState (compute)
//   - CommandList slot 37: SetComputeRoot32BitConstant
//   - CommandList slot 39: SetComputeRoot32BitConstants (32 constants, range checking, a write past the declared words is cut to the kept words)
//   - CommandList slot 45: SetComputeRootUnorderedAccessView (placed subrange offset handling, reject check)
//   - CommandList slot 14: Dispatch(x, y, z)
//   - Correct execution on GPU, subrange write to offset 256 and sentinel preserved at offset 0.
//   - Restoring the compute root after a same-layout root wrote its own constants and UAV through it.
//   - Negative controls: Dispatch with 0 threads or without a compute pipeline.
//   - Zero debug layer errors and healthy desktop device.
//
// Exit codes: 0 pass, 77 no physical adapter, other values name the failing step.
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using Microsoft::WRL::ComPtr;
using u64 = std::uint64_t;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) BOOL x2w_d3d12_fixture_restore_compute_root(void*);

namespace {
template <typename Method>
Method slot(IUnknown* object, unsigned index)
{
    return reinterpret_cast<Method>((*reinterpret_cast<void***>(object))[index]);
}

template <typename T>
ComPtr<T> native_object(ID3D12Device* device, ID3D12Device* guest, IUnknown* object)
{
    constexpr GUID key{0x10a62244, 0x1615, 0x4953, {0xa2, 0x52, 0xbe, 0x42, 0x35, 0x27, 0xa0, 0x61}};
    guest->SetPrivateDataInterface(key, object);
    ComPtr<IUnknown> stored;
    UINT bytes = sizeof(IUnknown*);
    const HRESULT read = device->GetPrivateData(key, &bytes, stored.GetAddressOf());
    guest->SetPrivateDataInterface(key, nullptr);
    if (FAILED(read) || !stored) return nullptr;
    ComPtr<T> result;
    stored.As(&result);
    return result;
}

constexpr char shader_source[] = R"(
RWByteAddressBuffer data : register(u0);
cbuffer RootConstants : register(b0) {
    uint scale;
    uint bias;
    uint2 pad2;
    uint4 pad4[6];
    uint3 pad3;
    uint extra;
};

[numthreads(1, 1, 1)]
void cs_main(uint3 group : SV_GroupID) {
    uint val = scale * (10 + group.x) + bias + extra;
    data.Store(0, val);
}
)";

ComPtr<ID3DBlob> compile_cs()
{
    ComPtr<ID3DBlob> code, errors;
    const HRESULT hr = D3DCompile(shader_source, sizeof(shader_source) - 1, nullptr, nullptr, nullptr,
                                  "cs_main", "cs_5_0", 0, 0, &code, &errors);
    if (FAILED(hr) && errors) std::fprintf(stderr, "%s\n", static_cast<const char*>(errors->GetBufferPointer()));
    return SUCCEEDED(hr) ? code : nullptr;
}

std::atomic<unsigned long long> debug_error_count{0};
void WINAPI on_message(D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID, LPCSTR text, void*)
{
    if (severity > D3D12_MESSAGE_SEVERITY_ERROR) return;
    if (++debug_error_count <= 12) std::printf("debug layer: %.300s\n", text);
}

struct Probe {
    ComPtr<ID3D12Device> desktop;
    DWORD cookie = 0;
    ComPtr<ID3D12InfoQueue1> queue1;
    bool watch()
    {
        return SUCCEEDED(desktop.As(&queue1)) &&
               SUCCEEDED(queue1->RegisterMessageCallback(on_message, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &cookie));
    }
};

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* res, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
{
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = res;
    barrier.Transition.StateBefore = from;
    barrier.Transition.StateAfter = to;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    list->ResourceBarrier(1, &barrier);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) return 2;
    const bool warp = std::strcmp(argv[1], "warp") == 0;

    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;

    ComPtr<IDXGIAdapter> adapter;
    if (warp) {
        if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) return 3;
    } else {
        for (UINT n = 0;; ++n) {
            ComPtr<IDXGIAdapter1> candidate;
            const HRESULT hr = factory->EnumAdapterByGpuPreference(n, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate));
            if (hr == DXGI_ERROR_NOT_FOUND) return 77;
            if (FAILED(hr)) return 3;
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(candidate->GetDesc1(&desc))) return 3;
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
                adapter = candidate;
                break;
            }
        }
    }
    if (!adapter) return warp ? 1 : 77;

    DXGI_ADAPTER_DESC adapter_desc{};
    if (FAILED(adapter->GetDesc(&adapter_desc))) return 3;
    std::wprintf(L"adapter=%s\n", adapter_desc.Description);

    Probe probe;
    std::array<unsigned char, 64> parameters{};
    ComPtr<ID3D12Device> guest;
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device), reinterpret_cast<void**>(guest.GetAddressOf()))))
        return 4;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&probe.desktop)))) return 4;

    if (!probe.watch()) std::printf("no debug layer message callback: validation not counted\n");

    // 1. Create compute root signature:
    //    Param 0: UAV (u0, space 0)
    //    Param 1: 32-bit constants (32 DWORDs at b0, space 0)
    D3D12_ROOT_PARAMETER root_params[2]{};
    root_params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    root_params[0].Descriptor.ShaderRegister = 0;
    root_params[0].Descriptor.RegisterSpace = 0;
    root_params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    root_params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    root_params[1].Constants = {0, 0, 32};
    root_params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    const D3D12_ROOT_SIGNATURE_DESC root_desc{2, root_params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> serialized_root;
    if (FAILED(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized_root, nullptr)))
        return 5;

    ComPtr<ID3D12RootSignature> guest_root;
    using CreateRootFn = HRESULT (WINAPI*)(IUnknown*, UINT, const void*, SIZE_T, REFIID, void**);
    if (FAILED(slot<CreateRootFn>(guest.Get(), 16)(guest.Get(), 0, serialized_root->GetBufferPointer(),
                                                   serialized_root->GetBufferSize(), __uuidof(ID3D12RootSignature),
                                                   reinterpret_cast<void**>(guest_root.GetAddressOf()))))
        return 7;

    // 2. Create compute pipeline state
    const auto cs_bytecode = compile_cs();
    if (!cs_bytecode) return 8;

    struct StreamDesc {
        alignas(void*) D3D12_PIPELINE_STATE_SUBOBJECT_TYPE root_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE;
        ID3D12RootSignature* root = nullptr;
        alignas(void*) D3D12_PIPELINE_STATE_SUBOBJECT_TYPE cs_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS;
        D3D12_SHADER_BYTECODE code{};
    } stream{D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, guest_root.Get(),
             D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS, {cs_bytecode->GetBufferPointer(), cs_bytecode->GetBufferSize()}};
    const D3D12_PIPELINE_STATE_STREAM_DESC stream_desc{sizeof(stream), &stream};

    using CreateStreamFn = HRESULT (WINAPI*)(IUnknown*, const D3D12_PIPELINE_STATE_STREAM_DESC*, REFIID, void**);
    ComPtr<IUnknown> guest_pipeline;
    if (FAILED(slot<CreateStreamFn>(guest.Get(), 86)(guest.Get(), &stream_desc, __uuidof(ID3D12PipelineState),
                                                    reinterpret_cast<void**>(guest_pipeline.GetAddressOf()))))
        return 9;

    // 3. Create placed UAV buffer (512 bytes) via device slot 133
    void* guest_mem = VirtualAlloc(nullptr, 512, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!guest_mem) return 10;

    const D3D12_RESOURCE_DESC placed_desc{
        D3D12_RESOURCE_DIMENSION_BUFFER, 0, 512, 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    };
    using PlaceFn = HRESULT (WINAPI*)(void*, void*, const D3D12_RESOURCE_DESC*, UINT, const void*, const GUID*, void**);
    ComPtr<IUnknown> placed_wrapper;
    if (FAILED(slot<PlaceFn>(guest.Get(), 133)(guest.Get(), guest_mem, &placed_desc, D3D12_RESOURCE_STATE_COMMON,
                                              nullptr, &__uuidof(ID3D12Resource),
                                              reinterpret_cast<void**>(placed_wrapper.GetAddressOf()))))
        return 10;

    const auto placed_res = native_object<ID3D12Resource>(probe.desktop.Get(), guest.Get(), placed_wrapper.Get());
    if (!placed_res) return 10;

    // Create staging upload buffer and readback buffer (512 bytes each)
    D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
    D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    const D3D12_RESOURCE_DESC buffer_desc{
        D3D12_RESOURCE_DIMENSION_BUFFER, 0, 512, 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE
    };

    ComPtr<ID3D12Resource> init_buffer, readback_buffer;
    if (FAILED(probe.desktop->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &buffer_desc,
                                                     D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&init_buffer))) ||
        FAILED(probe.desktop->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer_desc,
                                                     D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback_buffer))))
        return 10;

    // Pre-populate buffer with sentinels at offsets 0, 128, 384 and initial pattern at offset 256
    void* init_map = nullptr;
    if (FAILED(init_buffer->Map(0, nullptr, &init_map))) return 10;
    std::memset(init_map, 0, 512);
    *reinterpret_cast<UINT*>(init_map) = 0x11111111u; // Sentinel before subrange
    *reinterpret_cast<UINT*>(reinterpret_cast<char*>(init_map) + 128) = 0x33333333u; // Sentinel pinning unset constants mutant
    *reinterpret_cast<UINT*>(reinterpret_cast<char*>(init_map) + 256) = 0x22222222u; // Initial value at subrange
    *reinterpret_cast<UINT*>(reinterpret_cast<char*>(init_map) + 384) = 0x44444444u; // Sentinel pinning unset UAV / alignment mutant
    *reinterpret_cast<UINT*>(reinterpret_cast<char*>(init_map) + 448) = 0x55555555u; // Target for the cut-write dispatch
    init_buffer->Unmap(0, nullptr);

    // 4. Create CommandQueue, CommandAllocator, CommandList, Fence
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence;
    if (FAILED(probe.desktop->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue))) ||
        FAILED(probe.desktop->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
        return 11;

    ComPtr<ID3D12CommandAllocator> allocator;
    if (FAILED(guest->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))))
        return 11;

    using CreateListFn = HRESULT (WINAPI*)(IUnknown*, const void*, IUnknown*, IUnknown*, REFIID, void**);
    const UINT list_desc[]{D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    ComPtr<IUnknown> guest_list;
    if (FAILED(slot<CreateListFn>(guest.Get(), 73)(guest.Get(), list_desc, allocator.Get(), nullptr,
                                                   __uuidof(ID3D12GraphicsCommandList),
                                                   reinterpret_cast<void**>(guest_list.GetAddressOf()))))
        return 11;

    using ResetFn = HRESULT (WINAPI*)(IUnknown*, IUnknown*, void*);
    if (FAILED(slot<ResetFn>(guest_list.Get(), 139)(guest_list.Get(), allocator.Get(), nullptr)))
        return 12;

    const auto desktop_list = native_object<ID3D12GraphicsCommandList>(probe.desktop.Get(), guest.Get(), guest_list.Get());
    if (!desktop_list) return 13;

    // Initialize placed buffer with sentinel and subrange pattern, then transition to UNORDERED_ACCESS
    transition(desktop_list.Get(), placed_res.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    desktop_list->CopyBufferRegion(placed_res.Get(), 0, init_buffer.Get(), 0, 512);
    transition(desktop_list.Get(), placed_res.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // 5. Negative controls:
    // Negative control 1: Dispatch with 0 threads -> safely ignored (not recorded or skipped)
    using DispatchFn = void (WINAPI*)(IUnknown*, UINT, UINT, UINT);
    slot<DispatchFn>(guest_list.Get(), 14)(guest_list.Get(), 0, 1, 1);

    // Negative control 2: Dispatch without a compute pipeline -> counted as skipped
    slot<DispatchFn>(guest_list.Get(), 14)(guest_list.Get(), 1, 1, 1);

    // 6. Bind compute pipeline and root signature
    // Slot 25: SetPipelineState
    using PipelineFn = void (WINAPI*)(IUnknown*, IUnknown*);
    slot<PipelineFn>(guest_list.Get(), 25)(guest_list.Get(), guest_pipeline.Get());

    // Slot 33: SetComputeRootSignature
    using RootFn = void (WINAPI*)(IUnknown*, IUnknown*);
    slot<RootFn>(guest_list.Get(), 33)(guest_list.Get(), guest_root.Get());

    // Negative control 3: Dispatch with pipeline & root bound, but NO root arguments bound
    slot<DispatchFn>(guest_list.Get(), 14)(guest_list.Get(), 1, 1, 1);

    // 7. Test Root UAV (Slot 45) controls and PIN MUTANT: unset constants
    using UavFn = void (WINAPI*)(IUnknown*, UINT, D3D12_GPU_VIRTUAL_ADDRESS);
    const u64 guest_base_va = reinterpret_cast<u64>(guest_mem);
    const u64 guest_subrange_va_128 = guest_base_va + 128;
    const u64 guest_subrange_va_256 = guest_base_va + 256;
    const u64 guest_subrange_va_384 = guest_base_va + 384;

    // Control UAV-A: Bind valid UAV pointing to +128 on param 0, but constants (param 1) are UNSET
    // If a mutant relaxes the constants check, this dispatch executes and corrupts sentinel 0x33333333 at +128!
    slot<UavFn>(guest_list.Get(), 45)(guest_list.Get(), 0, guest_subrange_va_128);
    slot<DispatchFn>(guest_list.Get(), 14)(guest_list.Get(), 1, 1, 1);

    // Control UAV-B: Slot 45 with wrong parameter type (param 1 is 32-bit constants, not UAV) -> dropped
    slot<UavFn>(guest_list.Get(), 45)(guest_list.Get(), 1, guest_subrange_va_256);

    // Control UAV-C: Slot 45 with out-of-range parameter (param 2, root has 2 parameters) -> dropped
    slot<UavFn>(guest_list.Get(), 45)(guest_list.Get(), 2, guest_subrange_va_256);

    // Control UAV-Param16: Slot 45 with parameter 16 (parameter >= 16 guard) -> dropped
    slot<UavFn>(guest_list.Get(), 45)(guest_list.Get(), 16, guest_subrange_va_256);

    // Control UAV-Unplaced: Slot 45 with unplaced host memory -> not bound as GPU address, clears binding
    void* unplaced_mem = VirtualAlloc(nullptr, 65536, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (unplaced_mem) {
        slot<UavFn>(guest_list.Get(), 45)(guest_list.Get(), 0, reinterpret_cast<u64>(unplaced_mem));
        VirtualFree(unplaced_mem, 0, MEM_RELEASE);
    }
    slot<DispatchFn>(guest_list.Get(), 14)(guest_list.Get(), 1, 1, 1);

    // 8. Test Root Constants (32 constants, range checking, out-of-range write controls):
    using ConstantsFn = void (WINAPI*)(IUnknown*, UINT, UINT, const void*, UINT);
    using ConstantFn = void (WINAPI*)(IUnknown*, UINT, UINT, UINT);
    std::vector<UINT> constants_32(33, 0);
    constants_32[0] = 3;  // scale = 3
    constants_32[1] = 7;  // bias = 7
    constants_32[31] = 2; // extra = 2 (in DWORD 31, offset 124)

    // Control Const-A: Slot 39/37 on wrong parameter type (param 0 is UAV, not constants) -> dropped
    slot<ConstantsFn>(guest_list.Get(), 39)(guest_list.Get(), 0, 1, constants_32.data(), 0);
    slot<ConstantFn>(guest_list.Get(), 37)(guest_list.Get(), 0, 5, 0);

    // Control Const-B: Slot 39/37 on out-of-range parameter (param 2) -> dropped
    slot<ConstantsFn>(guest_list.Get(), 39)(guest_list.Get(), 2, 1, constants_32.data(), 0);
    slot<ConstantFn>(guest_list.Get(), 37)(guest_list.Get(), 2, 5, 0);

    // Control Const-Param16: Slot 39/37 on parameter 16 -> dropped
    slot<ConstantsFn>(guest_list.Get(), 39)(guest_list.Get(), 16, 1, constants_32.data(), 0);
    slot<ConstantFn>(guest_list.Get(), 37)(guest_list.Get(), 16, 5, 0);

    // Out-of-range negative controls on parameter 1:
    // Control Const-C: count exceeds parameter bound (count 33) -> cut to the 32 words both sides keep.
    // Bind a UAV with its own target first: this Dispatch must see the same 32 words on the desktop list, so a
    // store that marks all 32 words bound while the forward is skipped would write 0 here instead of 39.
    const u64 guest_subrange_va_448 = guest_base_va + 448;
    slot<UavFn>(guest_list.Get(), 45)(guest_list.Get(), 0, guest_subrange_va_448);
    slot<ConstantsFn>(guest_list.Get(), 39)(guest_list.Get(), 1, 33, constants_32.data(), 0);
    slot<DispatchFn>(guest_list.Get(), 14)(guest_list.Get(), 1, 1, 1);
    // Control Const-D: offset exceeds parameter bound (offset 32) -> rejected
    slot<ConstantsFn>(guest_list.Get(), 39)(guest_list.Get(), 1, 1, constants_32.data(), 32);
    slot<ConstantFn>(guest_list.Get(), 37)(guest_list.Get(), 1, 999, 32);
    // Control Const-E: offset + count exceeds parameter bound (offset 31, count 2) -> rejected
    slot<ConstantsFn>(guest_list.Get(), 39)(guest_list.Get(), 1, 2, constants_32.data(), 31);

    // Valid write of full 32 constants via Slot 39
    slot<ConstantsFn>(guest_list.Get(), 39)(guest_list.Get(), 1, 32, constants_32.data(), 0);

    // Slot 37: SetComputeRoot32BitConstant(param=1, data=5, offset=0) -> updates scale to 5!
    slot<ConstantFn>(guest_list.Get(), 37)(guest_list.Get(), 1, 5, 0);

    // 9. PIN MUTANTS: Unset UAV, alignment, and placed.reject
    // Bind valid UAV pointing to +384. Desktop list now holds UAV at +384!
    slot<UavFn>(guest_list.Get(), 45)(guest_list.Get(), 0, guest_subrange_va_384);

    // Control UAV-Misaligned: Slot 45 with misaligned address (+1 byte). Clears binding in pass,
    // but desktop list retains previous UAV at +384.
    slot<UavFn>(guest_list.Get(), 45)(guest_list.Get(), 0, guest_subrange_va_384 + 1);
    // Constants ARE bound. UAV is UNSET.
    // If a mutant relaxes the UAV check, this dispatch executes and corrupts sentinel 0x44444444 at +384!
    slot<DispatchFn>(guest_list.Get(), 14)(guest_list.Get(), 1, 1, 1);

    // Rebind UAV at +384
    slot<UavFn>(guest_list.Get(), 45)(guest_list.Get(), 0, guest_subrange_va_384);
    // Control UAV-Reject: Slot 45 with rejected placed range (~0u64). Clears binding in pass.
    slot<UavFn>(guest_list.Get(), 45)(guest_list.Get(), 0, ~u64{0});
    // If a mutant ignores placed.reject, this dispatch executes and corrupts sentinel 0x44444444 at +384!
    slot<DispatchFn>(guest_list.Get(), 14)(guest_list.Get(), 1, 1, 1);

    // 10. Valid bindings & test restore_compute_root:
    // Bind valid nonzero aligned subrange UAV at offset 256
    slot<UavFn>(guest_list.Get(), 45)(guest_list.Get(), 0, guest_subrange_va_256);

    // Simulate an intervening shim compute pass: it sets a root with the same layout and writes its own arguments
    // through it (the indexed-draw validation writes 8 constants through the root it sets). Both the constants and the
    // UAV are clobbered here, so a dispatch without the restore replay reads 0 at the wrong subrange.
    ComPtr<ID3D12RootSignature> clobber_root;
    if (FAILED(probe.desktop->CreateRootSignature(0, serialized_root->GetBufferPointer(), serialized_root->GetBufferSize(),
                                                  IID_PPV_ARGS(&clobber_root)))) return 13;
    desktop_list->SetComputeRootSignature(clobber_root.Get());
    const UINT zero_constants[32]{};
    desktop_list->SetComputeRoot32BitConstants(1, 32, zero_constants, 0);
    desktop_list->SetComputeRootUnorderedAccessView(0, placed_res->GetGPUVirtualAddress() + 384);

    // Restore the compute root signature, UAV, and all 32 root constants
    if (!x2w_d3d12_fixture_restore_compute_root(guest_list.Get())) return 13;

    // 11. Execute Dispatch: Slot 14 Dispatch(1, 1, 1) -> all root arguments restored!
    slot<DispatchFn>(guest_list.Get(), 14)(guest_list.Get(), 1, 1, 1);

    // 12. Copy placed resource to readback buffer
    transition(desktop_list.Get(), placed_res.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    desktop_list->CopyBufferRegion(readback_buffer.Get(), 0, placed_res.Get(), 0, 512);

    // Close guest list (calls Close on desktop_list)
    using CloseFn = HRESULT (WINAPI*)(IUnknown*);
    if (FAILED(slot<CloseFn>(guest_list.Get(), 9)(guest_list.Get()))) return 14;

    // Submit and synchronize
    ID3D12CommandList* submitted[]{desktop_list.Get()};
    queue->ExecuteCommandLists(1, submitted);
    if (FAILED(queue->Signal(fence.Get(), 1))) return 15;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (fence->GetCompletedValue() < 1) {
        fence->SetEventOnCompletion(1, event);
        WaitForSingleObject(event, 5000);
    }
    CloseHandle(event);

    // 13. Read back result and verify:
    // Expected val = scale * (10 + group.x) + bias + extra = 5 * (10 + 0) + 7 + 2 = 59 written at subrange offset 256.
    // Sentinel at offset 0 must remain 0x11111111.
    // Sentinel at offset 128 must remain 0x33333333 (pins unset constants mutant).
    // Sentinel at offset 384 must remain 0x44444444 (pins unset UAV / alignment mutant).
    void* mapped = nullptr;
    const D3D12_RANGE read_range{0, 512};
    if (FAILED(readback_buffer->Map(0, &read_range, &mapped))) return 16;
    UINT sentinel_0 = 0;
    UINT sentinel_128 = 0;
    UINT result_256 = 0;
    UINT sentinel_384 = 0;
    UINT result_448 = 0;
    std::memcpy(&sentinel_0, mapped, sizeof(UINT));
    std::memcpy(&sentinel_128, reinterpret_cast<const char*>(mapped) + 128, sizeof(UINT));
    std::memcpy(&result_256, reinterpret_cast<const char*>(mapped) + 256, sizeof(UINT));
    std::memcpy(&sentinel_384, reinterpret_cast<const char*>(mapped) + 384, sizeof(UINT));
    std::memcpy(&result_448, reinterpret_cast<const char*>(mapped) + 448, sizeof(UINT));
    readback_buffer->Unmap(0, nullptr);

    std::printf("sentinel at offset 0: 0x%08x (expected 0x11111111)\n", sentinel_0);
    std::printf("sentinel at offset 128: 0x%08x (expected 0x33333333, pins unset constants mutant)\n", sentinel_128);
    std::printf("subrange value at offset 256: %u (expected 59, scale=5, bias=7, extra=2)\n", result_256);
    std::printf("sentinel at offset 384: 0x%08x (expected 0x44444444, pins unset UAV / alignment mutant)\n", sentinel_384);
    std::printf("cut-write value at offset 448: %u (expected 39, scale=3, bias=7, extra=2)\n", result_448);
    if (sentinel_0 != 0x11111111u) return 17;
    if (sentinel_128 != 0x33333333u) return 18;
    if (result_256 != 59) return 19;
    if (sentinel_384 != 0x44444444u) return 20;
    if (result_448 != 39) return 21;


    // 13. Verify device health and debug layer
    const HRESULT removed = probe.desktop->GetDeviceRemovedReason();
    const unsigned long long errors = debug_error_count.load();
    std::printf("device removed reason: 0x%08lx, debug layer errors: %llu\n", static_cast<unsigned long>(removed),
                static_cast<unsigned long long>(errors));

    // Release wrappers and free memory
    placed_wrapper.Reset();
    VirtualFree(guest_mem, 0, MEM_RELEASE);

    return removed == S_OK && errors == 0 ? 0 : 70;
}
