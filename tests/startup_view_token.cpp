// The view token note_view writes into title descriptor memory (X2W_D3D12_VIEW_TOKEN_GUARDED, d3d12_capture.inl).
// Three descriptor heaps over title memory the fixture owns, protected after creation: writable, read-only, and with a
// guard page. A texture SRV (device slot 18) goes into each, and a second one into the writable heap. In both modes the
// writable heap gets both tokens, the read-only heap stays zero without a crash, and the guard page keeps its guard bit
// and stays zero. Two more heaps start 16 bytes before a page boundary (title backing has no alignment check), so
// their first descriptor's token crosses into a read-only or a guard page: in both modes the writable prefix keeps its
// bytes, the left page stays read-write without a guard bit, and the right page keeps its protection and stays zero.
// The stand-in's counters (x2w_d3d12_view_token_writes) tell the two paths apart: (0, 0) when the switch is unset,
// (2 written, 2 refused) when it is set; page-crossing tokens take the checked path and are not counted.
//
//   startup_view_token <warp|hardware> <unset|set>
//     (X2W_D3D12_BUDGET=1 X2W_D3D12_PERMISSIVE=1 X2W_D3D12_RENDER=1, plus X2W_D3D12_VIEW_TOKEN_GUARDED=1 for "set")
#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

using Microsoft::WRL::ComPtr;

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) void x2w_d3d12_view_token_writes(LONG* written, LONG* refused);

namespace {

constexpr std::uint32_t token_magic = 0x56573258;  // "X2WV"
constexpr UINT descriptors = 4;
constexpr SIZE_T region_bytes = 0x10000;
constexpr SIZE_T page = 0x1000;
constexpr SIZE_T prefix = 16;  // bytes of a crossing heap's first descriptor on the left page
constexpr unsigned char sentinel = 0xA5;

template<class Result = HRESULT, class... Args>
Result invoke(IUnknown* object, unsigned slot, Args... args)
{
    using Call = Result (WINAPI*)(IUnknown*, Args...);
    return reinterpret_cast<Call>((*reinterpret_cast<void***>(object))[slot])(object, args...);
}

int failures;

void expect(bool ok, const char* what)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

ComPtr<IDXGIAdapter1> pick_adapter(IDXGIFactory4* factory, bool warp)
{
    ComPtr<IDXGIAdapter1> adapter;
    if (warp) {
        factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter));
        return adapter;
    }
    wchar_t name[128]{};
    GetEnvironmentVariableW(L"XBOX2WINDOWS_ADAPTER", name, static_cast<DWORD>(std::size(name)));
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; ++i, adapter.Reset()) {
        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(adapter->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            (!name[0] || std::wcsstr(desc.Description, name)) &&
            SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr)))
            return adapter;
    }
    return nullptr;
}

DWORD protection(const void* address)
{
    MEMORY_BASIC_INFORMATION info{};
    return VirtualQuery(address, &info, sizeof(info)) ? info.Protect : 0;
}

bool all_zero(const void* memory, std::size_t bytes)
{
    const auto* p = static_cast<const unsigned char*>(memory);
    for (std::size_t i = 0; i < bytes; ++i)
        if (p[i]) return false;
    return true;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 3 || (std::strcmp(argv[1], "warp") && std::strcmp(argv[1], "hardware")) ||
        (std::strcmp(argv[2], "unset") && std::strcmp(argv[2], "set")))
        return 2;
    const bool guarded = std::strcmp(argv[2], "set") == 0;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (guarded != (GetEnvironmentVariableW(L"X2W_D3D12_VIEW_TOKEN_GUARDED", nullptr, 0) != 0)) {
        std::puts("FAIL: X2W_D3D12_VIEW_TOKEN_GUARDED does not match the mode argument");
        return 2;
    }

    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    const auto adapter = pick_adapter(factory.Get(), std::strcmp(argv[1], "warp") == 0);
    if (!adapter) return 77;
    DXGI_ADAPTER_DESC1 adapter_desc{};
    if (SUCCEEDED(adapter->GetDesc1(&adapter_desc))) std::printf("adapter=%ls\n", adapter_desc.Description);
    ComPtr<IUnknown> guest;
    std::array<unsigned char, 64> parameters{};
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device),
                                     reinterpret_cast<void**>(guest.GetAddressOf()))))
        return 3;
    const UINT stride = invoke<UINT>(guest.Get(), 15, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    if (!stride || UINT64{stride} * descriptors > 0x1000) return 3;

    // Title memory the fixture owns: one region per heap. The first three heaps start at their region's first page; the
    // crossing heaps start `prefix` bytes before the region's second page.
    enum { writable, read_only, guard, cross_read_only, cross_guard, heap_count };
    void* memory[heap_count]{};
    unsigned char* backing[heap_count]{};
    ComPtr<IUnknown> heaps[heap_count];
    for (int i = 0; i < heap_count; ++i) {
        memory[i] = VirtualAlloc(nullptr, region_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (!memory[i]) return 3;
        backing[i] = static_cast<unsigned char*>(memory[i]) + (i >= cross_read_only ? page - prefix : 0);
        const struct {
            D3D12_DESCRIPTOR_HEAP_DESC desc;
            UINT64 backing, bytes;
        } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, descriptors}, reinterpret_cast<UINT64>(backing[i]),
                    UINT64{stride} * descriptors};
        if (FAILED(invoke(guest.Get(), 14, &heap_desc, &__uuidof(ID3D12DescriptorHeap), heaps[i].GetAddressOf()))) return 3;
        if (invoke<UINT64>(heaps[i].Get(), 9) != reinterpret_cast<UINT64>(backing[i])) {
            std::puts("FAIL: the heap does not use the title memory it was given");
            return 3;
        }
    }
    for (int i = cross_read_only; i < heap_count; ++i) std::memset(backing[i], sentinel, prefix);
    DWORD old = 0;
    if (!VirtualProtect(memory[read_only], page, PAGE_READONLY, &old) ||
        !VirtualProtect(memory[guard], page, PAGE_READWRITE | PAGE_GUARD, &old) ||
        !VirtualProtect(backing[cross_read_only] + prefix, page, PAGE_READONLY, &old) ||
        !VirtualProtect(backing[cross_guard] + prefix, page, PAGE_READWRITE | PAGE_GUARD, &old))
        return 3;

    D3D12_HEAP_PROPERTIES default_heap{};
    default_heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = texture.Height = 16;
    texture.DepthOrArraySize = texture.MipLevels = 1;
    texture.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texture.SampleDesc.Count = 1;
    ComPtr<IUnknown> textures[2];
    const IID iid = __uuidof(ID3D12Resource);
    for (auto& t : textures)
        if (FAILED(invoke(guest.Get(), 27, &default_heap, D3D12_HEAP_FLAG_NONE, &texture, D3D12_RESOURCE_STATE_COMMON,
                          static_cast<const D3D12_CLEAR_VALUE*>(nullptr), &iid,
                          reinterpret_cast<void**>(t.GetAddressOf()))))
            return 3;
    // Every byte of the description identifies the view, so it is zeroed before its fields are set.
    D3D12_SHADER_RESOURCE_VIEW_DESC srv;
    std::memset(&srv, 0, sizeof(srv));
    srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    const auto view = [&](int texture_index, void* base, UINT index) {
        invoke<void>(guest.Get(), 18, static_cast<void*>(textures[texture_index].Get()), &srv,
                     reinterpret_cast<UINT64>(base) + UINT64{stride} * index);
    };
    view(0, memory[writable], 0);
    view(0, memory[read_only], 0);
    view(1, memory[guard], 0);
    view(1, memory[writable], 1);
    view(0, backing[cross_read_only], 0);
    view(1, backing[cross_guard], 0);

    const auto* words = static_cast<const std::uint32_t*>(memory[writable]);
    expect(words[0] == token_magic && words[2] == 18, "the writable heap's first descriptor holds no slot 18 token");
    const auto* second = reinterpret_cast<const std::uint32_t*>(static_cast<const unsigned char*>(memory[writable]) + stride);
    expect(second[0] == token_magic && second[1] != words[1], "the writable heap's second descriptor holds no new token");
    expect(all_zero(memory[read_only], UINT64{stride} * descriptors), "the read-only heap was written");
    MEMORY_BASIC_INFORMATION info{};
    expect(VirtualQuery(memory[guard], &info, sizeof(info)) && (info.Protect & PAGE_GUARD),
           "the guard page lost its guard bit");
    if (VirtualProtect(memory[guard], page, PAGE_READWRITE, &old))
        expect(all_zero(memory[guard], UINT64{stride} * descriptors), "the guard page was written");

    // The crossing tokens: nothing written on either side, and neither page's protection changed.
    for (int i = cross_read_only; i < heap_count; ++i) {
        const char* const name = i == cross_read_only ? "read-only" : "guard";
        const DWORD left = protection(backing[i]), right = protection(backing[i] + prefix);
        std::printf("crossing heap (%s right page): left page 0x%lx, right page 0x%lx\n", name, left, right);
        bool prefix_kept = true;
        for (SIZE_T b = 0; b < prefix; ++b) prefix_kept = prefix_kept && backing[i][b] == sentinel;
        expect(prefix_kept, "a crossing heap's writable prefix changed");
        expect(left == PAGE_READWRITE, "a crossing heap's left page changed its protection");
        expect(right == (i == cross_read_only ? PAGE_READONLY : (PAGE_READWRITE | PAGE_GUARD)),
               "a crossing heap's right page changed its protection");
        if (VirtualProtect(backing[i] + prefix, page, PAGE_READWRITE, &old))
            expect(all_zero(backing[i] + prefix, UINT64{stride} * descriptors - prefix),
                   "a crossing heap was written on its right page");
    }

    LONG written = -1, refused = -1;
    x2w_d3d12_view_token_writes(&written, &refused);
    std::printf("view tokens through the guard: written %ld, refused %ld\n", written, refused);
    if (guarded)
        expect(written == 2 && refused == 2, "the guarded path did not write 2 tokens and refuse 2");
    else
        expect(written == 0 && refused == 0, "the guarded path ran with the switch unset");
    if (failures) return 1;
    std::printf("PASS (%s): writable heap 2 tokens, read-only heap untouched, guard page kept, crossing tokens "
                "refused without a write or a protection change\n", argv[2]);
    return 0;
}
