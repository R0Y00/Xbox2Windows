#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
using Microsoft::WRL::ComPtr;

void check(HRESULT hr, const char* stage)
{
    if (SUCCEEDED(hr)) return;
    std::fprintf(stderr, "%s failed: 0x%08lx\n", stage, static_cast<unsigned long>(hr));
    std::exit(1);
}

template <typename Method>
Method slot(IUnknown* object, unsigned index)
{
    return reinterpret_cast<Method>((*reinterpret_cast<void***>(object))[index]);
}

using CreateBuilder = HRESULT(WINAPI*)(IUnknown*, const D3D12_COMPUTE_PIPELINE_STATE_DESC*, REFIID, void**);
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 3) return 2;
    const bool builder = std::strcmp(argv[2], "builder") == 0;
    if (!builder && std::strcmp(argv[2], "nonbuilder") != 0) return 2;

    SetEnvironmentVariableA("X2W_COMPUTE_BUILDER_GATE", "1");
    SetEnvironmentVariableA("X2W_D3D12_PERMISSIVE", "1");
    SetEnvironmentVariableA("X2W_D3D12_LIST_NOOP", "1");
    SetEnvironmentVariableA("X2W_D3D12_RENDER", "1");

    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> warp;
    check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "DXGI factory");
    check(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp)), "WARP adapter");

    ComPtr<ID3D12Device> guest;
    const std::uint64_t create_params[8]{};
    check(D3D12XboxCreateDevice(warp.Get(), create_params, IID_PPV_ARGS(&guest)), "guest device");

    // A console container: slot 62 accepts it as a placeholder, and the builder gate must refuse to
    // create a desktop PSO from it. The bytes are only read for the refusal, never submitted.
    static const unsigned char console_code[64] = {'D', 'X', 'B', 'C'};
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.CS = {console_code, sizeof(console_code)};

    using Slot62 = HRESULT(WINAPI*)(IUnknown*, const D3D12_COMPUTE_PIPELINE_STATE_DESC*, UINT,
                                    const void*, REFIID, void**);
    ComPtr<ID3D12PipelineState> pipeline;
    if (builder) {
        const std::wstring path(argv[1], argv[1] + std::strlen(argv[1]));
        HMODULE module = LoadLibraryW(path.c_str());
        if (!module) return 3;
        const auto create = reinterpret_cast<CreateBuilder>(GetProcAddress(module, "CreateBuilderPipeline"));
        if (!create) return 4;
        check(create(guest.Get(), &desc, IID_PPV_ARGS(&pipeline)), "builder pipeline");
        ComPtr<ID3D12PipelineState> second;
        check(create(guest.Get(), &desc, IID_PPV_ARGS(&second)), "second builder pipeline");
    } else {
        check(slot<Slot62>(guest.Get(), 62)(guest.Get(), &desc, 0, nullptr, IID_PPV_ARGS(&pipeline)),
              "non-builder pipeline");
    }

    ComPtr<ID3D12CommandAllocator> allocator;
    check(guest->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "allocator");
    const UINT list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    using CreateList = HRESULT(WINAPI*)(IUnknown*, const void*, IUnknown*, IUnknown*, REFIID, void**);
    ComPtr<ID3D12GraphicsCommandList> list;
    check(slot<CreateList>(guest.Get(), 73)(guest.Get(), list_desc, allocator.Get(), nullptr, IID_PPV_ARGS(&list)),
          "list");
    using Reset = HRESULT(WINAPI*)(IUnknown*, IUnknown*, void*, void*);
    check(slot<Reset>(list.Get(), 139)(list.Get(), allocator.Get(), nullptr, nullptr), "reset");

    using SetPipeline = void(WINAPI*)(IUnknown*, IUnknown*);
    using Dispatch = void(WINAPI*)(IUnknown*, UINT, UINT, UINT);
    slot<SetPipeline>(list.Get(), 25)(list.Get(), pipeline.Get());
    slot<Dispatch>(list.Get(), 14)(list.Get(), 1, 1, 1);

    D3D12_INDIRECT_ARGUMENT_DESC argument{};
    argument.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
    D3D12_COMMAND_SIGNATURE_DESC signature_desc{};
    signature_desc.ByteStride = sizeof(D3D12_DISPATCH_ARGUMENTS);
    signature_desc.NumArgumentDescs = 1;
    signature_desc.pArgumentDescs = &argument;
    using CreateSignature = HRESULT(WINAPI*)(IUnknown*, const D3D12_COMMAND_SIGNATURE_DESC*, IUnknown*, REFIID,
                                             void**);
    ComPtr<ID3D12CommandSignature> signature;
    check(slot<CreateSignature>(guest.Get(), 41)(guest.Get(), &signature_desc, nullptr, IID_PPV_ARGS(&signature)),
          "signature");

    using Execute = void(WINAPI*)(IUnknown*, IUnknown*, UINT, IUnknown*, UINT64, IUnknown*, UINT64);
    slot<Execute>(list.Get(), 61)(list.Get(), signature.Get(), 1, nullptr, 0, nullptr, 0);
    std::printf("%s done\n", argv[2]);
    return 0;
}
