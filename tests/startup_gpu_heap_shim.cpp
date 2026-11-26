// WP5 through the shim (X2W_GPU_ABI=1 X2W_D3D12_RENDER=1, WARP): a shader-visible view heap over title memory
// gets a mirror heap (gpu_heap_created), a view call into it is mirrored (gpu_heap_view), and a record the
// title copies there as bytes is found at the next submission (gpu_heap_submit). Read from the shim's log, which
// goes to standard error: here a file.
//
//   startup_gpu_heap_shim [--mutate]   --mutate: verify that a non-shader-visible heap has no mirror
//
// Exit codes: 0 pass, 1 a check failed, other values name the failing step.
#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
using Word = std::uint64_t;
template <class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }
Word invoke(void* object, unsigned index, Word a = 0, Word b = 0, Word c = 0, Word d = 0, Word e = 0, Word f = 0)
{
    using Slot = Word(WINAPI*)(void*, Word, Word, Word, Word, Word, Word);
    return reinterpret_cast<Slot>((*static_cast<void***>(object))[index])(object, a, b, c, d, e, f);
}
}  // namespace

int main(int argc, char** argv)
{
    const bool mutate = argc > 1 && std::strcmp(argv[1], "--mutate") == 0;
    const bool native = argc > 1 && std::strcmp(argv[1], "--native") == 0;
    const bool lazy = argc > 1 && std::strcmp(argv[1], "--lazy") == 0;
    char temp[MAX_PATH], log_path[MAX_PATH];
    GetTempPathA(MAX_PATH, temp);
    std::snprintf(log_path, sizeof(log_path), "%sx2w_gpu_heap_shim_%lu.log", temp, GetCurrentProcessId());
    HANDLE log_file = CreateFileA(log_path, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, CREATE_ALWAYS,
                                  FILE_ATTRIBUTE_NORMAL, nullptr);
    if (log_file == INVALID_HANDLE_VALUE) return 2;
    HANDLE old_error = GetStdHandle(STD_ERROR_HANDLE);
    SetStdHandle(STD_ERROR_HANDLE, log_file);

    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter> adapter;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) return 3;
    unsigned char parameters[64]{};
    void* device = nullptr;
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters, __uuidof(ID3D12Device), &device))) return 4;

    // Slot 14: a shader-visible view heap of 64 over title memory (32-byte records).
    constexpr UINT slots = 64;
    const UINT step = native ? static_cast<UINT>(invoke(device, 15, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)) : 32;
    std::vector<unsigned char> title(slots * step);
    struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        std::uint64_t extra[2];
    } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, slots,
                 mutate ? D3D12_DESCRIPTOR_HEAP_FLAG_NONE : D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0},
                {reinterpret_cast<std::uint64_t>(title.data()), title.size()}};
    void* heap = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(device, 14, arg(&heap_desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&heap)))) || !heap)
        return 5;

    // Slot 18: a null-resource texture SRV at title slot 3 (X2W writes its token there), then the title copies the
    // record to slot 9 as bytes.
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    invoke(device, 18, 0, arg(&srv), reinterpret_cast<Word>(title.data()) + 3 * step);
    std::memcpy(title.data() + 9 * step, title.data() + 3 * step, step);

    unsigned char* row_pages = nullptr;
    std::vector<unsigned char> staging(slots * step);
    std::vector<unsigned char> texels(64 * 65536, 0x5a);
    if (lazy || native) {
        heap_desc.desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        heap_desc.extra[0] = reinterpret_cast<Word>(staging.data());
        void* cpu_heap{};
        if (FAILED(static_cast<HRESULT>(invoke(device, 14, arg(&heap_desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&cpu_heap))))) return 10;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = desc.Height = 32;
        desc.MipLevels = desc.DepthOrArraySize = 1;
        desc.SampleDesc.Count = 1;
        desc.Format = srv.Format;
        desc.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x105);
        desc.Flags = static_cast<D3D12_RESOURCE_FLAGS>(0x300000);
        struct { D3D12_SHADER_RESOURCE_VIEW_DESC view; Word address; } input{srv};
        for (UINT i = 0; i < 32; ++i) {
            input.address = arg(texels.data() + i * 65536);
            invoke(device, 70, arg(&desc), arg(&input), arg(staging.data() + i * step));
            UINT tag;
            std::memcpy(&tag, staging.data() + i * step, sizeof(tag));
            if (tag != (native ? 0x56573258u : 0x4c573258u)) return 11;
        }
        if (native) {
            row_pages = static_cast<unsigned char*>(VirtualAlloc(nullptr, 12288, MEM_RESERVE, PAGE_NOACCESS));
            if (!row_pages || !VirtualAlloc(row_pages, 8192, MEM_COMMIT, PAGE_READWRITE)) return 12;
            desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
            desc.Format = input.view.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
            desc.Width = 1025;
            desc.Height = 1;
            desc.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x100);
            input.view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
            input.address = arg(row_pages + 8192 - desc.Width * 4);
            invoke(device, 70, arg(&desc), arg(&input), arg(staging.data() + 41 * step));
            UINT tag{};
            std::memcpy(&tag, staging.data() + 41 * step, sizeof(tag));
            if (tag != 0x56573258u) return 13;
        }
        // One CPU-only API copy, one promotion via API, another via byte copy.
        invoke(device, 24, 1, arg(staging.data() + 40 * step), arg(staging.data() + step), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        invoke(device, 24, 1, arg(title.data() + 20 * step), arg(staging.data() + 40 * step), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        std::memcpy(title.data() + 21 * step, staging.data() + 2 * step, step);
    }

    // Record and submit an empty title command list. Slot 73 creates a closed list.
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    const unsigned list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    void* queue = nullptr;
    void* allocator = nullptr;
    void* list = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(device, 8, arg(&queue_desc), arg(&__uuidof(ID3D12CommandQueue)), arg(&queue)))) || !queue)
        return 6;
    if (FAILED(static_cast<HRESULT>(invoke(device, 9, 0, arg(&__uuidof(ID3D12CommandAllocator)), arg(&allocator)))) || !allocator)
        return 7;
    if (FAILED(static_cast<HRESULT>(invoke(device, 73, arg(list_desc), arg(allocator), 0, arg(&__uuidof(ID3D12GraphicsCommandList)), arg(&list)))) ||
        !list)
        return 8;
    if (FAILED(static_cast<HRESULT>(invoke(list, 139, arg(allocator))))) return 9;
    if (FAILED(static_cast<HRESULT>(invoke(list, 9)))) return 10;
    void* lists[] = {list};
    invoke(queue, 10, 1, arg(lists));

    SetStdHandle(STD_ERROR_HANDLE, old_error);
    CloseHandle(log_file);
    std::string text;
    if (FILE* f = std::fopen(log_path, "rb")) {
        char buffer[4096];
        for (std::size_t n; (n = std::fread(buffer, 1, sizeof(buffer), f)) > 0;) text.append(buffer, n);
        std::fclose(f);
    }
    DeleteFileA(log_path);
    int failures = 0;
    const auto check = [&](bool ok, const char* what) {
        std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
        if (!ok) ++failures;
    };
    const auto line_of = [&](const char* start) {
        const auto at = text.find(start);
        if (at == std::string::npos) return std::string();
        return text.substr(at, text.find('\n', at) - at);
    };
    const std::string mirror = line_of("x2w-gpu heap: mirror of the title's view heap");
    const std::string submit = line_of("x2w-gpu heap submit 1:");
    std::printf("%s\n%s\n", mirror.c_str(), submit.c_str());
    if (native) {
        VirtualFree(row_pages, 0, MEM_RELEASE);
        check(!line_of("x2w-memsrv form 1d:").empty(), "1025-texel row ends at the last readable byte");
        check(mirror.empty(), "native descriptor handling has no translated heap");
        check(line_of("x2w-gpu detiled texture descriptors staged").empty(), "native reads do not use deferred descriptors");
        check(!line_of("x2w-memsrv detiled memory SRV at").empty(), "native detiled texture was created");
        return failures ? 1 : 0;
    }
    check(!line_of("x2w-gpu submits 1:").empty(), "the shim submitted one command list");
    if (lazy) {
        check(!line_of("x2w-gpu detiled texture descriptors staged 32").empty(), "32 CPU staging writes were deferred");
        const std::string prefix = "x2w-memsrv linear texture at ";
        size_t cursor = 0, allocations = 0;
        while ((cursor = text.find(prefix, cursor)) != std::string::npos) { ++allocations; cursor += prefix.size(); }
        check(allocations == 0, "creating and copying descriptors without a GPU consumer commits no texture");
    }
    if (mutate) {
        check(mirror.empty() && submit.empty(), "the non-shader-visible heap has no mirror to scan");
    } else {
        check(!mirror.empty() && mirror.find(" 64 title slots") != std::string::npos, "a mirror heap of the title's 64 slots");
        check(submit.find(lazy ? "submit 1: 2 records changed" : "submit 1: 1 records changed") != std::string::npos, "submit 1 found the record copied as bytes");
    }
    if (failures) std::printf("startup_gpu_heap_shim: %d failures\n", failures);
    else std::printf("startup_gpu_heap_shim: pass\n");
    return failures ? 1 : 0;
}
