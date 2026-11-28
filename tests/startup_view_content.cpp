// Content view tokens (X2W_D3D12_VIEW_CONTENT_TOKENS=1, prototype; view_content_token.h, d3d12_capture.inl). A buffer
// SRV, a CBV and a texture SRV go into a descriptor heap over title memory the fixture owns. The fixture keeps copies
// of their tokens in plain memory, as a title copies descriptors with its own memory copy, and copies the buffer SRV
// into a second descriptor with CopyDescriptorsSimple. Then 5000 other buffer SRVs are created into the first
// descriptor. In both modes every copy still resolves to the view it was made from, through the token readers'
// path (token_view) and through the second descriptor's shadow slot. With the switch on, the buffer views' records
// are freed and reused, so the record count stays bounded; with it off, it grows by one per view.
//
//   startup_view_content <warp|hardware> <off|on>
//     (X2W_D3D12_BUDGET=1 X2W_D3D12_PERMISSIVE=1 X2W_D3D12_RENDER=1, plus X2W_D3D12_VIEW_CONTENT_TOKENS=1 for "on")
#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

using Microsoft::WRL::ComPtr;

struct ViewStoreProbe {
    unsigned long long records, materialized, ids, records_capacity, materialized_capacity;
    unsigned shadow, view_refs, retire_when_idle;
};
struct ResolvedViewProbe {
    unsigned known, slot, desc_bytes, materialized;
    unsigned long long resource;
    unsigned char desc[64];
};
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) void x2w_d3d12_view_probe(unsigned long long, void*, ViewStoreProbe*);
extern "C" __declspec(dllimport) void x2w_d3d12_resolve_view(const void*, unsigned long long, ResolvedViewProbe*);
extern "C" __declspec(dllimport) void x2w_d3d12_view_content_counts(long long*, long long*, long long*, long long*);
extern "C" __declspec(dllimport) void x2w_d3d12_set_view_parked_keep(unsigned);

namespace {

constexpr std::uint32_t token_magic = 0x56573258;  // "X2WV"
constexpr std::uint32_t content_bit = 0x80000000u;
constexpr UINT descriptors = 8;
constexpr UINT overwrites = 5000;

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

D3D12_SHADER_RESOURCE_VIEW_DESC raw_window(UINT64 first, UINT count)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC srv;
    std::memset(&srv, 0, sizeof(srv));
    srv.Format = DXGI_FORMAT_R32_TYPELESS;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.FirstElement = first;
    srv.Buffer.NumElements = count;
    srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    return srv;
}

struct Token {
    std::uint32_t words[8];
};

// The view a token copy or a descriptor's shadow slot names must be exactly `slot`, `resource` and `desc`.
// A CBV is recorded as not created (the stand-in keeps it title-side); every SRV here must be created.
template<class Desc>
void expect_view(const ResolvedViewProbe& got, unsigned slot, const void* resource, const Desc& desc, const char* what)
{
    const unsigned created = slot == 17 ? 0u : 1u;
    const bool same = got.known && got.slot == slot && got.resource == reinterpret_cast<unsigned long long>(resource) &&
                      got.desc_bytes == sizeof(desc) && !std::memcmp(got.desc, &desc, sizeof(desc)) &&
                      got.materialized == created;
    if (!same)
        std::printf("  %s: known %u slot %u resource 0x%llx desc_bytes %u created %u\n", what, got.known, got.slot,
                    got.resource, got.desc_bytes, got.materialized);
    expect(same, what);
}

ResolvedViewProbe resolve(const Token& token)
{
    ResolvedViewProbe out{};
    x2w_d3d12_resolve_view(token.words, 0, &out);
    return out;
}
ResolvedViewProbe resolve_shadow(UINT64 handle)
{
    ResolvedViewProbe out{};
    x2w_d3d12_resolve_view(nullptr, handle, &out);
    return out;
}
ViewStoreProbe probe(UINT64 handle)
{
    ViewStoreProbe p{};
    x2w_d3d12_view_probe(handle, nullptr, &p);
    return p;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 3 || (std::strcmp(argv[1], "warp") && std::strcmp(argv[1], "hardware")) ||
        (std::strcmp(argv[2], "off") && std::strcmp(argv[2], "on")))
        return 2;
    const bool on = std::strcmp(argv[2], "on") == 0;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    wchar_t value[4]{};
    const DWORD length = GetEnvironmentVariableW(L"X2W_D3D12_VIEW_CONTENT_TOKENS", value, 4);
    if (on != (length == 1 && value[0] == L'1')) {
        std::puts("FAIL: X2W_D3D12_VIEW_CONTENT_TOKENS does not match the mode argument");
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
    // No parked records kept back, so the 5000 views reach the reuse (the stand-in keeps 65536).
    x2w_d3d12_set_view_parked_keep(0);
    const UINT stride = invoke<UINT>(guest.Get(), 15, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    if (!stride || UINT64{stride} * descriptors > 0x1000) return 3;

    void* memory = VirtualAlloc(nullptr, 0x1000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!memory) return 3;
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        UINT64 backing, bytes;
    } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, descriptors}, reinterpret_cast<UINT64>(memory),
                UINT64{stride} * descriptors};
    ComPtr<IUnknown> heap;
    if (FAILED(invoke(guest.Get(), 14, &heap_desc, &__uuidof(ID3D12DescriptorHeap), heap.GetAddressOf()))) return 3;
    const auto handle = [&](UINT index) { return reinterpret_cast<UINT64>(memory) + UINT64{stride} * index; };
    const auto token_at = [&](UINT index) {
        Token t;
        std::memcpy(t.words, reinterpret_cast<const void*>(handle(index)), sizeof(t.words));
        return t;
    };

    D3D12_HEAP_PROPERTIES default_heap{};
    default_heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC buffer_desc{};
    buffer_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer_desc.Width = 0x40000;
    buffer_desc.Height = buffer_desc.DepthOrArraySize = buffer_desc.MipLevels = 1;
    buffer_desc.SampleDesc.Count = 1;
    buffer_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_RESOURCE_DESC texture_desc{};
    texture_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture_desc.Width = texture_desc.Height = 16;
    texture_desc.DepthOrArraySize = texture_desc.MipLevels = 1;
    texture_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texture_desc.SampleDesc.Count = 1;
    const IID iid = __uuidof(ID3D12Resource);
    ComPtr<IUnknown> buffer, texture;
    if (FAILED(invoke(guest.Get(), 27, &default_heap, D3D12_HEAP_FLAG_NONE, &buffer_desc, D3D12_RESOURCE_STATE_COMMON,
                      static_cast<const D3D12_CLEAR_VALUE*>(nullptr), &iid, reinterpret_cast<void**>(buffer.GetAddressOf()))) ||
        FAILED(invoke(guest.Get(), 27, &default_heap, D3D12_HEAP_FLAG_NONE, &texture_desc, D3D12_RESOURCE_STATE_COMMON,
                      static_cast<const D3D12_CLEAR_VALUE*>(nullptr), &iid, reinterpret_cast<void**>(texture.GetAddressOf()))))
        return 3;
    const auto srv = [&](IUnknown* resource, const D3D12_SHADER_RESOURCE_VIEW_DESC& desc, UINT index) {
        invoke<void>(guest.Get(), 18, static_cast<void*>(resource), &desc, handle(index));
    };

    // 0: a buffer SRV, 1: a CBV, 2: a texture SRV; title-side copies of their tokens; 3: a shim-side copy of 0.
    const D3D12_SHADER_RESOURCE_VIEW_DESC first = raw_window(0, 16);
    srv(buffer.Get(), first, 0);
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbv{};
    cbv.BufferLocation = 0x00000123456789a0ull;
    cbv.SizeInBytes = 256;
    invoke<void>(guest.Get(), 17, &cbv, handle(1));
    D3D12_SHADER_RESOURCE_VIEW_DESC texture_view;
    std::memset(&texture_view, 0, sizeof(texture_view));
    texture_view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texture_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    texture_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    texture_view.Texture2D.MipLevels = 1;
    srv(texture.Get(), texture_view, 2);
    const Token buffer_copy = token_at(0), cbv_copy = token_at(1), texture_copy = token_at(2);
    invoke<void>(guest.Get(), 24, 1u, handle(3), handle(0), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    for (const Token* t : {&buffer_copy, &cbv_copy, &texture_copy})
        expect(t->words[0] == token_magic, "a descriptor holds no view token");
    expect(((buffer_copy.words[1] & content_bit) != 0) == on, "the buffer SRV token's kind does not match the mode");
    expect(((cbv_copy.words[1] & content_bit) != 0) == on, "the CBV token's kind does not match the mode");
    expect(!(texture_copy.words[1] & content_bit), "the texture SRV got a content token");
    expect(!std::memcmp(token_at(3).words, buffer_copy.words, sizeof(Token)), "the copied descriptor holds another token");
    const unsigned long long records_before = probe(handle(0)).records;
    long long written0 = 0, freed0 = 0, reused0 = 0, revived0 = 0;
    x2w_d3d12_view_content_counts(&written0, &freed0, &reused0, &revived0);

    for (UINT k = 1; k <= overwrites; ++k) srv(buffer.Get(), raw_window(UINT64{k} * 4, 16), 0);
    const unsigned long long records_after = probe(handle(0)).records;
    long long written = 0, freed = 0, reused = 0, revived = 0;
    x2w_d3d12_view_content_counts(&written, &freed, &reused, &revived);
    std::printf("records %llu -> %llu over %u buffer views; content tokens written %lld, records freed %lld, reused %lld\n",
                records_before, records_after, overwrites, written - written0, freed - freed0, reused - reused0);
    if (on) {
        expect(records_after <= records_before + 8, "the records grew with content tokens");
        expect(written - written0 >= overwrites && freed - freed0 >= overwrites - 1 && reused - reused0 >= overwrites - 2,
               "the buffer views' records were not freed and reused");
    } else {
        expect(records_after >= records_before + overwrites, "the records did not grow by one per view without the switch");
        expect(written == 0 && freed == 0 && reused == 0, "content tokens or freed records without the switch");
    }

    // Every copy still names the view it was made from.
    expect_view(resolve(buffer_copy), 18, buffer.Get(), first, "the title-side copy of the buffer SRV token");
    expect_view(resolve(cbv_copy), 17, nullptr, cbv, "the title-side copy of the CBV token");
    expect_view(resolve(texture_copy), 18, texture.Get(), texture_view, "the title-side copy of the texture SRV token");
    expect_view(resolve_shadow(handle(3)), 18, buffer.Get(), first, "the copied descriptor's shadow slot");
    expect_view(resolve_shadow(handle(0)), 18, buffer.Get(), raw_window(UINT64{overwrites} * 4, 16),
                "the overwritten descriptor's shadow slot");
    expect_view(resolve(token_at(0)), 18, buffer.Get(), raw_window(UINT64{overwrites} * 4, 16),
                "the overwritten descriptor's token");

    // The last shadow slot naming the first view lets it go; its title-side copy still resolves.
    srv(buffer.Get(), raw_window(8, 8), 3);
    long long freed_last = 0;
    x2w_d3d12_view_content_counts(&written, &freed_last, &reused, &revived);
    if (on) expect(freed_last == freed + 1, "the first view's record was not freed with its last shadow slot");
    expect_view(resolve(buffer_copy), 18, buffer.Get(), first, "the title-side copy after the last shadow slot let go");

    // The same view created again while its record is parked finds that record: no new record, one revival.
    const unsigned long long records_parked = probe(handle(0)).records;
    long long revived_before = 0, reused_before = 0;
    x2w_d3d12_view_content_counts(&written, &freed, &reused_before, &revived_before);
    srv(buffer.Get(), first, 6);
    x2w_d3d12_view_content_counts(&written, &freed, &reused, &revived);
    expect_view(resolve_shadow(handle(6)), 18, buffer.Get(), first, "the re-created first view's shadow slot");
    expect_view(resolve(token_at(6)), 18, buffer.Get(), first, "the re-created first view's token");
    std::printf("re-created first view: revived %lld -> %lld, reused %lld -> %lld, records %llu -> %llu\n", revived_before,
                revived, reused_before, reused, records_parked, probe(handle(0)).records);
    if (on)
        expect(revived == revived_before + 1 && reused == reused_before && probe(handle(0)).records == records_parked,
               "the re-created first view did not revive its parked record");

    // Padding bytes do not make another content token.
    D3D12_SHADER_RESOURCE_VIEW_DESC padded = raw_window(40000, 16);
    std::memset(reinterpret_cast<unsigned char*>(&padded) + 12, 0xcd, 4);
    srv(buffer.Get(), padded, 4);
    srv(buffer.Get(), raw_window(40000, 16), 5);
    expect(!std::memcmp(token_at(4).words, token_at(5).words, sizeof(Token)) == on,
           "views that differ only in padding bytes: content tokens differ, or id tokens match");

    // The first view's record is parked again and then reused for other views (the fixture keeps none back); the
    // title-side copy of its token must still name it.
    srv(buffer.Get(), raw_window(16, 8), 6);
    for (UINT k = 0; k < 4; ++k) srv(buffer.Get(), raw_window(UINT64{k} * 4 + 32, 4), 0);
    expect_view(resolve(buffer_copy), 18, buffer.Get(), first, "the title-side copy after its record was reused");

    if (failures) return 1;
    std::printf("PASS (%s)\n", argv[2]);
    return 0;
}
