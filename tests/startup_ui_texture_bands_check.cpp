// Stage-by-stage trace and regression for horizontal alternating nonzero/black row bands
// on textured indexed UI draws (#3132).
//
// Usage: startup_ui_texture_bands_check warp|hardware
// Exit codes: 0 pass, 77 adapter unavailable, other nonzero test failure.
#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using Microsoft::WRL::ComPtr;

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) HRESULT WINAPI XGCreateTextureComputer(const D3D12_RESOURCE_DESC*, void***, UINT64, UINT64);
extern "C" __declspec(dllimport) void x2w_d3d12_list_copy_texture_stats(LONG* forwarded, LONG* refused);
extern "C" __declspec(dllimport) void x2w_d3d12_texture_written(void* base, unsigned long long bytes);

namespace {
using Word = std::uint64_t;
template <class T>
Word arg(T* value) { return reinterpret_cast<Word>(value); }

Word invoke(void* object, unsigned index, Word a = 0, Word b = 0, Word c = 0, Word d = 0,
            Word e = 0, Word f = 0, Word g = 0, Word h = 0, Word i = 0, Word j = 0)
{
    using Slot = Word (WINAPI*)(void*, Word, Word, Word, Word, Word, Word, Word, Word, Word, Word);
    return reinterpret_cast<Slot>((*static_cast<void***>(object))[index])(object, a, b, c, d, e, f, g, h, i, j);
}

bool ok(Word hr) { return SUCCEEDED(static_cast<HRESULT>(hr)); }

constexpr UINT kWidth = 32;
constexpr UINT kHeight = 32;
constexpr UINT kMips = 3;

struct UiVertex {
    float x, y;
    float u, v;
};

struct Rgba {
    std::uint8_t r, g, b, a;
    bool operator==(const Rgba& o) const { return r == o.r && g == o.g && b == o.b && a == o.a; }
    bool operator!=(const Rgba& o) const { return !(*this == o); }
    bool is_black() const { return r == 0 && g == 0 && b == 0 && a == 0; }
};

Rgba expected_texel(UINT mip, UINT x, UINT y)
{
    return {
        static_cast<std::uint8_t>(24 + y * 6),
        static_cast<std::uint8_t>(32 + x * 6),
        static_cast<std::uint8_t>(64 + mip * 80),
        255,
    };
}

Rgba expected_gradient(UINT x, UINT y)
{
    return {
        static_cast<std::uint8_t>(16 + x * 6),
        static_cast<std::uint8_t>(16 + y * 6),
        128,
        255,
    };
}

std::uint64_t fnv_bytes(const void* data, std::size_t bytes, std::uint64_t hash = 1469598103934665603ull)
{
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) hash = (hash ^ p[i]) * 1099511628211ull;
    return hash;
}

std::uint64_t shader_key(const void* vs, std::size_t vs_bytes, const void* ps, std::size_t ps_bytes,
                         const void* root, std::size_t root_bytes)
{
    constexpr char rev[] = "x2w-startup-shader-29";
    auto h = fnv_bytes(rev, sizeof(rev) - 1);
    h = fnv_bytes(vs, vs_bytes, h);
    h = fnv_bytes(ps, ps_bytes, h);
    return fnv_bytes(root, root_bytes, h);
}

bool write_file_bytes(const wchar_t* path, const void* data, std::size_t bytes)
{
    HANDLE f = CreateFileW(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) return false;
    DWORD written = 0;
    const bool done = WriteFile(f, data, static_cast<DWORD>(bytes), &written, nullptr) && written == bytes;
    CloseHandle(f);
    return done;
}

bool compile_hlsl(const char* source, const char* target, ComPtr<ID3DBlob>& blob)
{
    ComPtr<ID3DBlob> errors;
    const HRESULT hr = D3DCompile(source, std::strlen(source), nullptr, nullptr, nullptr, "main", target,
                                  D3DCOMPILE_OPTIMIZATION_LEVEL2, 0, &blob, &errors);
    if (FAILED(hr)) {
        std::fprintf(stderr, "D3DCompile(%s) failed 0x%08lx: %s\n", target, static_cast<unsigned long>(hr),
                     errors ? static_cast<const char*>(errors->GetBufferPointer()) : "");
        return false;
    }
    return true;
}

bool publish_translated_pso(void* device, const wchar_t* cache_dir, const char* vs_hlsl, const char* ps_hlsl,
                            const char* manifest, void* root, const void* root_blob, std::size_t root_bytes,
                            void*& out_pso)
{
    ComPtr<ID3DBlob> vs, ps;
    if (!compile_hlsl(vs_hlsl, "vs_5_1", vs) || !compile_hlsl(ps_hlsl, "ps_5_1", ps)) return false;

    const auto key = shader_key(vs->GetBufferPointer(), vs->GetBufferSize(),
                                ps->GetBufferPointer(), ps->GetBufferSize(),
                                root_blob, root_bytes);
    wchar_t vs_path[MAX_PATH], ps_path[MAX_PATH], txt_path[MAX_PATH];
    std::swprintf(vs_path, MAX_PATH, L"%s\\%016llx.vs.cso", cache_dir, static_cast<unsigned long long>(key));
    std::swprintf(ps_path, MAX_PATH, L"%s\\%016llx.ps.cso", cache_dir, static_cast<unsigned long long>(key));
    std::swprintf(txt_path, MAX_PATH, L"%s\\%016llx.txt", cache_dir, static_cast<unsigned long long>(key));
    if (!write_file_bytes(vs_path, vs->GetBufferPointer(), vs->GetBufferSize()) ||
        !write_file_bytes(ps_path, ps->GetBufferPointer(), ps->GetBufferSize()) ||
        !write_file_bytes(txt_path, manifest, std::strlen(manifest))) {
        return false;
    }

    struct alignas(8) Stage { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE tag; D3D12_SHADER_BYTECODE code; };
    struct alignas(8) RootSub { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE tag; void* root; };
    struct Stream { Stage ps; RootSub root; Stage vs; } stream{
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, {ps->GetBufferPointer(), ps->GetBufferSize()}},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, {vs->GetBufferPointer(), vs->GetBufferSize()}},
    };
    const D3D12_PIPELINE_STATE_STREAM_DESC stream_desc{sizeof(stream), &stream};
    return ok(invoke(device, 86, arg(&stream_desc), arg(&__uuidof(ID3D12PipelineState)), arg(&out_pso))) && out_pso;
}

// Clean-room GFX10 thin non-XOR tiler for 32-bit texels (modes 0x105 = 4K_S, 0x10a = 64K_D)
// matching AMD addrlib GFX10 pattern tables (Mesa 25.1.0, MIT).
struct Gfx10Tiler {
    UINT width = 0, height = 0, mips = 1, mode = 0x105;
    UINT block_w = 0, block_h = 0, block_bytes = 0, first_tail = 0;
    std::array<UINT, 15> start{}, blocks_wide{}, ox{}, oy{};
    std::array<UINT, 256> x_bits{}, y_bits{};
    UINT total_bytes = 0;

    void init(UINT w, UINT h, UINT m, UINT layout_mode)
    {
        width = w; height = h; mips = m; mode = layout_mode;
        const UINT log2 = (layout_mode & 0xff) == 5 ? 12u : 16u;
        block_bytes = 1u << log2;
        const char* pat[16] = {};
        // For 4-byte elements, standard and display micro-block patterns are identical.
        const char* micro[8] = {"0", "0", "X0", "X1", "Y0", "Y1", "Y2", "X2"};
        UINT nx = 3, ny = 3;
        for (UINT b = 0; b < 8; ++b) pat[b] = micro[b];
        std::array<UINT, 8> x_at{}, y_at{};
        x_at[0] = 2; x_at[1] = 3; x_at[2] = 7;
        y_at[0] = 4; y_at[1] = 5; y_at[2] = 6;
        for (UINT b = 8; b < log2; ++b) {
            if (b & 1) x_at[nx++] = b;
            else y_at[ny++] = b;
        }
        block_w = 1u << nx;
        block_h = 1u << ny;
        for (UINT v = 0; v < 256; ++v) {
            x_bits[v] = y_bits[v] = 0;
            for (UINT bit = 0; bit < 8; ++bit) {
                if (bit < nx && ((v >> bit) & 1)) x_bits[v] |= 1u << x_at[bit];
                if (bit < ny && ((v >> bit) & 1)) y_bits[v] |= 1u << y_at[bit];
            }
        }
        const UINT tail_mips = log2 - 4;
        first_tail = mips;
        for (UINT mip = 0; mips > 1 && mip < mips; ++mip) {
            const UINT mw = (std::max)(1u, width >> mip);
            const UINT mh = (std::max)(1u, height >> mip);
            if (mw <= block_w / 2 && mh <= block_h && mips - mip <= tail_mips) {
                first_tail = mip;
                break;
            }
        }
        UINT off = first_tail < mips ? block_bytes : 0;
        for (UINT mip = first_tail; mip-- > 0;) {
            const UINT mw = (std::max)(1u, width >> mip);
            const UINT mh = (std::max)(1u, height >> mip);
            blocks_wide[mip] = (mw + block_w - 1) / block_w;
            start[mip] = off;
            off += blocks_wide[mip] * ((mh + block_h - 1) / block_h) * block_bytes;
        }
        total_bytes = off;
        for (UINT mip = first_tail; mip < mips; ++mip) {
            const UINT idx = tail_mips - 1 - (mip - first_tail);
            const UINT toff = idx > 6 ? 16u << idx : idx << 8;
            UINT tx = 0, ty = 0;
            for (UINT b = 0; b < 6; ++b) {
                ty |= ((toff >> (8 + 2 * b)) & 1) << b;
                tx |= ((toff >> (9 + 2 * b)) & 1) << b;
            }
            start[mip] = toff;
            ox[mip] = tx * 8;
            oy[mip] = ty * 8;
        }
    }

    UINT addr(UINT mip, UINT x, UINT y) const
    {
        if (mip >= first_tail) return x_bits[(ox[mip] + x) & 255] | y_bits[(oy[mip] + y) & 255];
        const UINT blk = (y / block_h) * blocks_wide[mip] + (x / block_w);
        return start[mip] + blk * block_bytes + (x_bits[x % block_w] | y_bits[y % block_h]);
    }
};

struct MemoryViewAt {
    D3D12_SHADER_RESOURCE_VIEW_DESC view;
    Word address;
    Word pad[2];
};

void print_platform_header(IDXGIAdapter* adapter)
{
    using RtlGetVersionFn = LONG(WINAPI*)(OSVERSIONINFOW*);
    OSVERSIONINFOW ver{sizeof(ver)};
    if (const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll")) {
        const auto fn = reinterpret_cast<RtlGetVersionFn>(GetProcAddress(ntdll, "RtlGetVersion"));
        if (fn) fn(&ver);
    }
    DXGI_ADAPTER_DESC desc{};
    LARGE_INTEGER driver{};
    adapter->GetDesc(&desc);
    adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driver);
    char dbg[16]{};
    const bool debug_env = GetEnvironmentVariableA("X2W_D3D12_DEBUG", dbg, sizeof(dbg)) && dbg[0] != '0';
    std::printf("platform: Windows %lu.%lu.%lu adapter=\"%ls\" driver=%u.%u.%u.%u x2w_debug_env=%s\n",
                ver.dwMajorVersion, ver.dwMinorVersion, ver.dwBuildNumber, desc.Description,
                HIWORD(driver.HighPart), LOWORD(driver.HighPart), HIWORD(driver.LowPart), LOWORD(driver.LowPart),
                debug_env ? "on" : "off");
}

int pick_adapter(const char* backend, ComPtr<IDXGIAdapter>& adapter)
{
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    if (std::strcmp(backend, "warp") == 0) {
        return SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))) ? 0 : 77;
    }
    if (std::strcmp(backend, "hardware") != 0) return 2;
    ComPtr<IDXGIFactory6> factory6;
    if (FAILED(factory.As(&factory6))) return 77;
    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> cand;
        if (factory6->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&cand)) == DXGI_ERROR_NOT_FOUND)
            return 77;
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(cand->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
        if (SUCCEEDED(D3D12CreateDevice(cand.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
            adapter = cand;
            return 0;
        }
    }
}

bool wait_fence(void* fence, UINT64 value)
{
    const auto deadline = GetTickCount64() + 10000;
    while (invoke(fence, 8) < value && GetTickCount64() < deadline) Sleep(1);
    return invoke(fence, 8) == value;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes)
{
    return {D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
            D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
}

void* make_committed(void* device, D3D12_HEAP_TYPE heap_type, const D3D12_RESOURCE_DESC& desc, D3D12_RESOURCE_STATES state)
{
    const D3D12_HEAP_PROPERTIES props{heap_type};
    void* out = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(device, 127, arg(&props), D3D12_HEAP_FLAG_NONE, arg(&desc), state,
                                            0, 0, 0, 0, arg(&__uuidof(ID3D12Resource)), arg(&out)))))
        return nullptr;
    return out;
}

void print_rows_summary(const char* tag, const Rgba* data, UINT pitch_texels, UINT w, UINT h, UINT max_rows = 6)
{
    std::printf("  %s (first %u rows at x=0,1):", tag, (std::min)(h, max_rows));
    for (UINT y = 0; y < (std::min)(h, max_rows); ++y) {
        const Rgba p0 = data[y * pitch_texels + 0];
        const Rgba p1 = w > 1 ? data[y * pitch_texels + 1] : p0;
        std::printf(" y%u=[(%u,%u,%u,%u),(%u,%u,%u,%u)]", y, p0.r, p0.g, p0.b, p0.a, p1.r, p1.g, p1.b, p1.a);
    }
    std::printf("\n");
}

bool find_first_texel_mismatch(const Rgba* actual, UINT pitch_texels, UINT mip, UINT w, UINT h,
                                UINT& bad_x, UINT& bad_y, Rgba& got, Rgba& exp)
{
    for (UINT y = 0; y < h; ++y) {
        for (UINT x = 0; x < w; ++x) {
            const Rgba g = actual[y * pitch_texels + x];
            const Rgba e = expected_texel(mip, x, y);
            if (g != e) {
                bad_x = x; bad_y = y; got = g; exp = e;
                return true;
            }
        }
    }
    return false;
}

bool has_alternating_black_bands(const Rgba* rows, UINT pitch_texels, UINT h)
{
    if (h < 4) return false;
    bool even_nz = true, odd_black = true;
    for (UINT y = 0; y < h; ++y) {
        const bool blk = rows[y * pitch_texels].is_black();
        if (y % 2 == 0) even_nz &= !blk;
        else odd_black &= blk;
    }
    return even_nz && odd_black;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: startup_ui_texture_bands_check warp|hardware\n");
        return 2;
    }
    SetEnvironmentVariableA("X2W_D3D12_PERMISSIVE", "1");
    SetEnvironmentVariableA("X2W_D3D12_LIST_NOOP", "1");
    SetEnvironmentVariableA("X2W_D3D12_RENDER", "1");
    if (!GetEnvironmentVariableA("X2W_D3D12_DEBUG", nullptr, 0))
        SetEnvironmentVariableA("X2W_D3D12_DEBUG", "1");

    wchar_t temp_root[MAX_PATH]{}, cache_dir[MAX_PATH]{};
    if (!GetTempPathW(MAX_PATH, temp_root)) return 3;
    std::swprintf(cache_dir, MAX_PATH, L"%sx2w_ui_bands_%lu", temp_root, GetCurrentProcessId());
    CreateDirectoryW(cache_dir, nullptr);
    SetEnvironmentVariableW(L"X2W_SHADER_CACHE", cache_dir);

    auto cleanup_cache = [&] {
        wchar_t pattern[MAX_PATH];
        std::swprintf(pattern, MAX_PATH, L"%s\\*", cache_dir);
        WIN32_FIND_DATAW fd{};
        HANDLE h = FindFirstFileW(pattern, &fd);
        if (h != INVALID_HANDLE_VALUE) {
            do {
                if (std::wcscmp(fd.cFileName, L".") != 0 && std::wcscmp(fd.cFileName, L"..") != 0) {
                    wchar_t f[MAX_PATH];
                    std::swprintf(f, MAX_PATH, L"%s\\%s", cache_dir, fd.cFileName);
                    DeleteFileW(f);
                }
            } while (FindNextFileW(h, &fd));
            FindClose(h);
        }
        RemoveDirectoryW(cache_dir);
    };

    ComPtr<IDXGIAdapter> adapter;
    const int adapter_rc = pick_adapter(argv[1], adapter);
    if (adapter_rc != 0) {
        cleanup_cache();
        return adapter_rc;
    }
    print_platform_header(adapter.Get());

    ComPtr<ID3D12Device> desktop;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&desktop)))) {
        cleanup_cache();
        return 77;
    }

    std::array<unsigned char, 64> params{};
    void* device = nullptr;
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), params.data(), __uuidof(ID3D12Device), &device)) || !device) {
        cleanup_cache();
        return 5;
    }

    // Guest root signature: [0] SRV descriptor table (1 entry at t1 for PS), matching runtime manifests.
    // Note: translated VS/PS use d3d12_draw's create_translated_root(b0 constants, t0.. SRV table, s0.. static samplers).
    const D3D12_DESCRIPTOR_RANGE srv_range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1, 0, 0};
    D3D12_ROOT_PARAMETER root_param{};
    root_param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_param.DescriptorTable = {1, &srv_range};
    root_param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    const D3D12_ROOT_SIGNATURE_DESC root_desc{1, &root_param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> root_blob;
    if (FAILED(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &root_blob, nullptr))) {
        cleanup_cache();
        return 6;
    }
    void* guest_root = nullptr;
    if (!ok(invoke(device, 16, 0, arg(root_blob->GetBufferPointer()), root_blob->GetBufferSize(),
                   arg(&__uuidof(ID3D12RootSignature)), arg(&guest_root))) || !guest_root) {
        cleanup_cache();
        return 7;
    }

    // 1) Preceding gradient draw PSO (indexed quad, no texture)
    const char* grad_vs = R"(
cbuffer Entry : register(b0) { uint basev; uint stride; };
ByteAddressBuffer vb : register(t0);
struct Out { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
Out main(uint vid : SV_VertexID) {
    uint off = (vid + basev) * stride;
    float2 p = asfloat(vb.Load2(off));
    float2 uv = asfloat(vb.Load2(off + 8));
    Out o; o.pos = float4(p, 0.0, 1.0); o.uv = uv; return o;
})";
    const char* grad_ps = R"(
struct In { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
float4 main(In i) : SV_Target {
    uint x = uint(i.pos.x);
    uint y = uint(i.pos.y);
    return float4(float(16 + x * 6) / 255.0, float(16 + y * 6) / 255.0, 128.0 / 255.0, 1.0);
})";
    const char* grad_manifest = "x2w-manifest 1\nud 2\nudsrc 0 basevertex\nudsrc 1 stride 0\nsrv 0 buf vb 0\n";

    // 2) Textured indexed UI quad PSO (vertex buffer t0 via vb 0, Texture2D t1 via table 0 0, point-clamp sampler s0)
    // w0 = 2|(2<<3)|(2<<6) = 0x92 -> CLAMP, filter=0 (POINT)
    const char* tex_ps = R"(
Texture2D<float4> uiTex : register(t1);
SamplerState uiSmp : register(s0);
struct In { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
float4 main(In i) : SV_Target {
    return uiTex.SampleLevel(uiSmp, i.uv, 0);
})";
    const char* tex_manifest = "x2w-manifest 1\nud 2\nudsrc 0 basevertex\nudsrc 1 stride 0\nsrv 0 buf vb 0\nsrv 1 tex2d table 0 0\nsmp 0 00000092 00000000 00000000 00000000\n";

    void *grad_pso = nullptr, *tex_pso = nullptr;
    if (!publish_translated_pso(device, cache_dir, grad_vs, grad_ps, grad_manifest, guest_root,
                                root_blob->GetBufferPointer(), root_blob->GetBufferSize(), grad_pso) ||
        !publish_translated_pso(device, cache_dir, grad_vs, tex_ps, tex_manifest, guest_root,
                                root_blob->GetBufferPointer(), root_blob->GetBufferSize(), tex_pso)) {
        cleanup_cache();
        return 8;
    }

    // Render target & readbacks (using desktop resources for RT via wrapped RTV + guest resources for upload/readback)
    const D3D12_RESOURCE_DESC rt_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, kWidth, kHeight, 1, 1,
                                      DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN,
                                      D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    void* rt = make_committed(device, D3D12_HEAP_TYPE_DEFAULT, rt_desc, D3D12_RESOURCE_STATE_COMMON);
    if (!rt) { cleanup_cache(); return 9; }

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT rt_fp{};
    UINT rt_rows = 0;
    UINT64 rt_row_bytes = 0, rt_total = 0;
    desktop->GetCopyableFootprints(&rt_desc, 0, 1, 0, &rt_fp, &rt_rows, &rt_row_bytes, &rt_total);
    void* rt_readback = make_committed(device, D3D12_HEAP_TYPE_READBACK, buffer_desc(rt_total), D3D12_RESOURCE_STATE_COPY_DEST);

    const struct { D3D12_DESCRIPTOR_HEAP_DESC d; Word b, s; } rtv_hdesc{{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1}, 0, 0};
    void* rtv_heap = nullptr;
    if (!ok(invoke(device, 14, arg(&rtv_hdesc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&rtv_heap))) || !rt_readback) {
        cleanup_cache(); return 10;
    }
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv{static_cast<SIZE_T>(invoke(rtv_heap, 9))};
    const D3D12_RENDER_TARGET_VIEW_DESC rtv_desc{DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RTV_DIMENSION_TEXTURE2D, {}};
    invoke(device, 20, arg(rt), arg(&rtv_desc), rtv.ptr);

    const UINT desc_inc = static_cast<UINT>(invoke(device, 15, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV));
    std::vector<unsigned char> view_shadow(16 * desc_inc, 0);
    const struct { D3D12_DESCRIPTOR_HEAP_DESC d; Word b, s; } srv_hdesc{
        {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 16, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE},
        arg(view_shadow.data()), view_shadow.size()};
    void* srv_heap = nullptr;
    if (!ok(invoke(device, 14, arg(&srv_hdesc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&srv_heap)))) {
        cleanup_cache(); return 11;
    }
    const Word srv_cpu0 = invoke(srv_heap, 9);
    const Word srv_gpu0 = invoke(srv_heap, 10);

    D3D12_COMMAND_QUEUE_DESC qd{};
    const unsigned list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    void *queue = nullptr, *alloc = nullptr, *list = nullptr, *fence = nullptr;
    if (!ok(invoke(device, 8, arg(&qd), arg(&__uuidof(ID3D12CommandQueue)), arg(&queue))) ||
        !ok(invoke(device, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, arg(&__uuidof(ID3D12CommandAllocator)), arg(&alloc))) ||
        !ok(invoke(device, 73, arg(list_desc), arg(alloc), 0, arg(&__uuidof(ID3D12GraphicsCommandList)), arg(&list))) ||
        !ok(invoke(device, 36, 0, 0, arg(&__uuidof(ID3D12Fence)), arg(&fence)))) {
        cleanup_cache(); return 12;
    }
    Word fence_val = 0;

    // Indexed quad vertices (full 32x32 target, top-down UV [0,1]x[0,1])
    static const UiVertex quad_verts[4] = {
        {-1.0f,  1.0f, 0.0f, 0.0f},
        { 1.0f,  1.0f, 1.0f, 0.0f},
        { 1.0f, -1.0f, 1.0f, 1.0f},
        {-1.0f, -1.0f, 0.0f, 1.0f},
    };
    static const std::uint16_t quad_indices[6] = {0, 1, 2, 0, 2, 3};
    const D3D12_VERTEX_BUFFER_VIEW vbv{reinterpret_cast<D3D12_GPU_VIRTUAL_ADDRESS>(quad_verts), sizeof(quad_verts), sizeof(UiVertex)};
    const D3D12_INDEX_BUFFER_VIEW ibv{reinterpret_cast<D3D12_GPU_VIRTUAL_ADDRESS>(quad_indices), sizeof(quad_indices), DXGI_FORMAT_R16_UINT};

    const D3D12_RESOURCE_DESC mip_desc_3{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, kWidth, kHeight, 1, kMips,
                                        DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN,
                                        D3D12_RESOURCE_FLAG_NONE};
    std::array<D3D12_PLACED_SUBRESOURCE_FOOTPRINT, kMips> mip_fp{};
    std::array<UINT, kMips> mip_rows{};
    std::array<UINT64, kMips> mip_row_bytes{};
    UINT64 mip_total = 0;
    desktop->GetCopyableFootprints(&mip_desc_3, 0, kMips, 0, mip_fp.data(), mip_rows.data(), mip_row_bytes.data(), &mip_total);

    std::printf("generated 32x32 3-mip RGBA8 source (mip0 row_bytes=%llu, desktop RowPitch=%u, total=%llu):\n",
                static_cast<unsigned long long>(mip_row_bytes[0]), mip_fp[0].Footprint.RowPitch,
                static_cast<unsigned long long>(mip_total));
    std::vector<Rgba> gen_preview(kWidth * 6);
    for (UINT y = 0; y < 6; ++y)
        for (UINT x = 0; x < kWidth; ++x) gen_preview[y * kWidth + x] = expected_texel(0, x, y);
    print_rows_summary("cpu mip0", gen_preview.data(), kWidth, kWidth, 6);

    auto readback_rt = [&](std::vector<Rgba>& out, UINT& pitch_texels) -> bool {
        out.assign(static_cast<std::size_t>(rt_total / 4), {0, 0, 0, 0});
        pitch_texels = rt_fp.Footprint.RowPitch / 4;
        if (!ok(invoke(alloc, 8)) || !ok(invoke(list, 139, arg(alloc), 0))) return false;
        D3D12_TEXTURE_COPY_LOCATION dst_loc{static_cast<ID3D12Resource*>(rt_readback), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        dst_loc.PlacedFootprint = rt_fp;
        D3D12_TEXTURE_COPY_LOCATION src_loc{static_cast<ID3D12Resource*>(rt), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        invoke(list, 16, arg(&dst_loc), 0, 0, 0, arg(&src_loc), 0);
        if (!ok(invoke(list, 9))) return false;
        void* lists[] = {list};
        invoke(queue, 10, 1, arg(lists));
        if (!ok(invoke(queue, 14, arg(fence), ++fence_val)) || !wait_fence(fence, fence_val)) return false;
        void* mapped = nullptr;
        if (!ok(invoke(rt_readback, 8, 0, 0, arg(&mapped))) || !mapped) return false;
        std::memcpy(out.data(), mapped, static_cast<std::size_t>(rt_total));
        invoke(rt_readback, 9, 0, 0);
        return true;
    };

    auto readback_texture_mips = [&](void* guest_tex, std::vector<std::vector<Rgba>>& mips_out) -> bool {
        void* rb = make_committed(device, D3D12_HEAP_TYPE_READBACK, buffer_desc(mip_total), D3D12_RESOURCE_STATE_COPY_DEST);
        if (!rb) return false;
        if (!ok(invoke(alloc, 8)) || !ok(invoke(list, 139, arg(alloc), 0))) { invoke(rb, 2); return false; }
        for (UINT m = 0; m < kMips; ++m) {
            D3D12_TEXTURE_COPY_LOCATION dst_loc{static_cast<ID3D12Resource*>(rb), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            dst_loc.PlacedFootprint = mip_fp[m];
            D3D12_TEXTURE_COPY_LOCATION src_loc{static_cast<ID3D12Resource*>(guest_tex), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            src_loc.SubresourceIndex = m;
            invoke(list, 16, arg(&dst_loc), 0, 0, 0, arg(&src_loc), 0);
        }
        if (!ok(invoke(list, 9))) { invoke(rb, 2); return false; }
        void* lists[] = {list};
        invoke(queue, 10, 1, arg(lists));
        if (!ok(invoke(queue, 14, arg(fence), ++fence_val)) || !wait_fence(fence, fence_val)) { invoke(rb, 2); return false; }
        void* mapped = nullptr;
        if (!ok(invoke(rb, 8, 0, 0, arg(&mapped))) || !mapped) { invoke(rb, 2); return false; }
        const auto* base = static_cast<const unsigned char*>(mapped);
        mips_out.resize(kMips);
        for (UINT m = 0; m < kMips; ++m) {
            const UINT pitch = mip_fp[m].Footprint.RowPitch / 4;
            mips_out[m].resize(static_cast<std::size_t>(pitch) * mip_rows[m]);
            std::memcpy(mips_out[m].data(), base + mip_fp[m].Offset, static_cast<std::size_t>(pitch) * 4 * mip_rows[m]);
        }
        invoke(rb, 9, 0, 0);
        invoke(rb, 2);
        return true;
    };

    auto draw_gradient_then_ui = [&](Word srv_gpu, const char* case_name, UINT view_mip) -> bool {
        if (!ok(invoke(alloc, 8)) || !ok(invoke(list, 139, arg(alloc), 0))) return false;
        const float clear[] = {0, 0, 0, 0};
        const D3D12_VIEWPORT vp{0, 0, static_cast<float>(kWidth), static_cast<float>(kHeight), 0, 1};
        const D3D12_RECT sc{0, 0, static_cast<LONG>(kWidth), static_cast<LONG>(kHeight)};
        invoke(list, 52, rtv.ptr, arg(clear));
        invoke(list, 50, 1, arg(&rtv));
        invoke(list, 21, 1, arg(&vp));
        invoke(list, 22, 1, arg(&sc));
        invoke(list, 20, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        invoke(list, 32, 1, arg(&srv_heap));
        invoke(list, 48, 0, 1, arg(&vbv));
        invoke(list, 47, arg(&ibv));
        // 1) Separate indexed gradient draw first
        invoke(list, 25, arg(grad_pso));
        invoke(list, 13, 6, 1, 0, 0, 0);
        if (!ok(invoke(list, 9))) return false;
        void* lists1[] = {list};
        invoke(queue, 10, 1, arg(lists1));
        if (!ok(invoke(queue, 14, arg(fence), ++fence_val)) || !wait_fence(fence, fence_val)) return false;

        std::vector<Rgba> pre_px;
        UINT pre_pitch = 0;
        if (!readback_rt(pre_px, pre_pitch)) return false;
        for (UINT y = 0; y < kHeight; ++y) {
            for (UINT x = 0; x < kWidth; ++x) {
                if (pre_px[y * pre_pitch + x] != expected_gradient(x, y)) {
                    std::fprintf(stderr, "FAIL %s: pre-draw gradient mismatch at (%u,%u)\n", case_name, x, y);
                    return false;
                }
            }
        }

        // 2) Textured indexed UI quad draw over the gradient
        if (!ok(invoke(alloc, 8)) || !ok(invoke(list, 139, arg(alloc), 0))) return false;
        invoke(list, 50, 1, arg(&rtv));
        invoke(list, 21, 1, arg(&vp));
        invoke(list, 22, 1, arg(&sc));
        invoke(list, 20, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        invoke(list, 32, 1, arg(&srv_heap));
        invoke(list, 36, 0, srv_gpu);
        invoke(list, 48, 0, 1, arg(&vbv));
        invoke(list, 47, arg(&ibv));
        invoke(list, 25, arg(tex_pso));
        invoke(list, 13, 6, 1, 0, 0, 0);
        if (!ok(invoke(list, 9))) return false;
        invoke(queue, 10, 1, arg(lists1));
        if (!ok(invoke(queue, 14, arg(fence), ++fence_val)) || !wait_fence(fence, fence_val)) return false;
        (void)view_mip;
        return true;
    };

    auto verify_rt_for_mip = [&](const char* name, UINT mip, bool expect_pass, bool expect_bands) -> bool {
        std::vector<Rgba> px;
        UINT pitch = 0;
        if (!readback_rt(px, pitch)) return false;
        const UINT mw = kWidth >> mip, mh = kHeight >> mip;
        // Print sample coordinates, UV and neighboring row RGBA
        const Rgba row0 = px[0 * pitch + 0];
        const Rgba row1 = px[1 * pitch + 0];
        const double u0 = 0.5 / kWidth;
        const double v0 = 0.5 / kHeight;
        const double v1 = 1.5 / kHeight;
        std::printf("  %s (view_mip=%u): uv(0,0)=(%.5f,%.5f) uv(0,1)=(%.5f,%.5f) final_rgba[y0]=(%u,%u,%u,%u) final_rgba[y1]=(%u,%u,%u,%u)\n",
                    name, mip, u0, v0, u0, v1, row0.r, row0.g, row0.b, row0.a, row1.r, row1.g, row1.b, row1.a);
        print_rows_summary("post-ui rt", px.data(), pitch, kWidth, kHeight);
        const bool banded = has_alternating_black_bands(px.data(), pitch, kHeight);
        UINT bad_x = 0, bad_y = 0;
        Rgba got{}, exp{};
        bool any_bad = false;
        for (UINT y = 0; y < kHeight; ++y) {
            const UINT sy = (y * mh) / kHeight;
            for (UINT x = 0; x < kWidth; ++x) {
                const UINT sx = (x * mw) / kWidth;
                const Rgba e = expected_texel(mip, sx, sy);
                const Rgba g = px[y * pitch + x];
                if (g != e) {
                    bad_x = x; bad_y = y; got = g; exp = e; any_bad = true;
                    break;
                }
            }
            if (any_bad) break;
        }
        if (any_bad) {
            std::printf("  %s (view_mip=%u): first sample mismatch at (%u,%u) got=(%u,%u,%u,%u) expected=(%u,%u,%u,%u) alternating_black_bands=%s\n",
                        name, mip, bad_x, bad_y, got.r, got.g, got.b, got.a, exp.r, exp.g, exp.b, exp.a,
                        banded ? "YES" : "NO");
        } else {
            std::printf("  %s (view_mip=%u): ALL %u pixels exact\n", name, mip, kWidth * kHeight);
        }
        if (expect_pass && any_bad) return false;
        if (expect_bands && (!any_bad || !banded)) return false;
        return true;
    };

    auto make_srv18 = [&](void* tex, UINT slot_idx, UINT most_detailed_mip, UINT mip_levels) -> Word {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MostDetailedMip = most_detailed_mip;
        srv.Texture2D.MipLevels = mip_levels;
        invoke(device, 18, arg(tex), arg(&srv), srv_cpu0 + Word{slot_idx} * desc_inc);
        return srv_gpu0 + Word{slot_idx} * desc_inc;
    };

    // Helper for filling a desktop-footprint buffer
    auto fill_linear_footprints = [&](unsigned char* dst) {
        for (UINT m = 0; m < kMips; ++m) {
            const UINT w = kWidth >> m, h = kHeight >> m;
            for (UINT y = 0; y < h; ++y) {
                for (UINT x = 0; x < w; ++x) {
                    const Rgba t = expected_texel(m, x, y);
                    std::memcpy(dst + mip_fp[m].Offset + y * mip_fp[m].Footprint.RowPitch + x * 4, &t, 4);
                }
            }
        }
    };

    // =========================================================================
    // PATH 1: Linear Reference (CopyTextureRegion list slot 16)
    // =========================================================================
    bool all_ok = true;
    int first_fail_rc = 0;
    auto record_fail = [&](int rc) {
        all_ok = false;
        if (!first_fail_rc) first_fail_rc = rc;
    };

    std::printf("[Path 1] Linear Reference (CopyTextureRegion slot 16 -> committed & placed DEFAULT textures)\n");
    {
        void* tex = make_committed(device, D3D12_HEAP_TYPE_DEFAULT, mip_desc_3, D3D12_RESOURCE_STATE_COMMON);
        void* up = make_committed(device, D3D12_HEAP_TYPE_UPLOAD, buffer_desc(mip_total), D3D12_RESOURCE_STATE_GENERIC_READ);
        if (!tex || !up) { cleanup_cache(); return 20; }
        void* mapped = nullptr;
        if (!ok(invoke(up, 8, 0, 0, arg(&mapped))) || !mapped) { cleanup_cache(); return 21; }
        std::memset(mapped, 0, static_cast<std::size_t>(mip_total));
        fill_linear_footprints(static_cast<unsigned char*>(mapped));
        invoke(up, 9, 0, 0);

        if (!ok(invoke(alloc, 8)) || !ok(invoke(list, 139, arg(alloc), 0))) { cleanup_cache(); return 22; }
        for (UINT m = 0; m < kMips; ++m) {
            D3D12_TEXTURE_COPY_LOCATION to{static_cast<ID3D12Resource*>(tex), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            to.SubresourceIndex = m;
            D3D12_TEXTURE_COPY_LOCATION from{static_cast<ID3D12Resource*>(up), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            from.PlacedFootprint = mip_fp[m];
            invoke(list, 16, arg(&to), 0, 0, 0, arg(&from), 0);
        }
        if (!ok(invoke(list, 9))) { cleanup_cache(); return 23; }
        void* lists[] = {list};
        invoke(queue, 10, 1, arg(lists));
        if (!ok(invoke(queue, 14, arg(fence), ++fence_val)) || !wait_fence(fence, fence_val)) { cleanup_cache(); return 24; }

        std::vector<std::vector<Rgba>> rb_mips;
        if (!readback_texture_mips(tex, rb_mips)) { cleanup_cache(); return 25; }
        print_rows_summary("gpu texels mip0", rb_mips[0].data(), mip_fp[0].Footprint.RowPitch / 4, kWidth, kHeight);
        for (UINT m = 0; m < kMips; ++m) {
            UINT bx = 0, by = 0; Rgba g{}, e{};
            if (find_first_texel_mismatch(rb_mips[m].data(), mip_fp[m].Footprint.RowPitch / 4, m, kWidth >> m, kHeight >> m, bx, by, g, e)) {
                std::fprintf(stderr, "FAIL Path 1: GPU texel mismatch mip%u (%u,%u)\n", m, bx, by);
                record_fail(26);
            }
        }
        for (UINT m = 0; m < kMips; ++m) {
            const Word gpu = make_srv18(tex, 0, m, kMips - m);
            if (!draw_gradient_then_ui(gpu, "Path 1", m) || !verify_rt_for_mip("Path 1", m, true, false)) {
                record_fail(27);
            }
        }
        invoke(up, 2);
        invoke(tex, 2);
    }

    // Path 1b: CopyTextureRegion (slot 16) into a slot-133 placed texture whose title_memory holds zeros,
    // then advance 16 frames + mutate title_memory to confirm GPU-written state is preserved across rechecks.
    std::printf("[Path 1b] CopyTextureRegion into slot-133 texture preserves GPU writes across title_memory mutation\n");
    {
        auto* backing = static_cast<unsigned char*>(VirtualAlloc(nullptr, static_cast<SIZE_T>(mip_total),
                                                               MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        std::memset(backing, 0, static_cast<std::size_t>(mip_total));
        void* tex = nullptr;
        if (!ok(invoke(device, 133, arg(backing), arg(&mip_desc_3), D3D12_RESOURCE_STATE_COMMON,
                       0, arg(&__uuidof(ID3D12Resource)), arg(&tex))) || !tex) {
            cleanup_cache(); return 28;
        }
        void* up = make_committed(device, D3D12_HEAP_TYPE_UPLOAD, buffer_desc(mip_total), D3D12_RESOURCE_STATE_GENERIC_READ);
        void* mapped = nullptr;
        if (!up || !ok(invoke(up, 8, 0, 0, arg(&mapped))) || !mapped) { cleanup_cache(); return 28; }
        fill_linear_footprints(static_cast<unsigned char*>(mapped));
        invoke(up, 9, 0, 0);

        if (!ok(invoke(alloc, 8)) || !ok(invoke(list, 139, arg(alloc), 0))) { cleanup_cache(); return 28; }
        for (UINT m = 0; m < kMips; ++m) {
            D3D12_TEXTURE_COPY_LOCATION to{static_cast<ID3D12Resource*>(tex), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            to.SubresourceIndex = m;
            D3D12_TEXTURE_COPY_LOCATION from{static_cast<ID3D12Resource*>(up), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            from.PlacedFootprint = mip_fp[m];
            invoke(list, 16, arg(&to), 0, 0, 0, arg(&from), 0);
        }
        if (!ok(invoke(list, 9))) { cleanup_cache(); return 28; }
        void* lists[] = {list};
        invoke(queue, 10, 1, arg(lists));
        if (!ok(invoke(queue, 14, arg(fence), ++fence_val)) || !wait_fence(fence, fence_val)) { cleanup_cache(); return 28; }

        // Mutate backing memory with distinct non-zero bytes and advance >15 frames so CPU recheck would
        // overwrite the CopyTextureRegion result unless list_copy_texture marked the texture GPU-written.
        std::memset(backing, 0x5A, static_cast<std::size_t>(mip_total));
        for (int f = 0; f < 16; ++f) invoke(queue, 32, 0, 0, 0);
        const Word gpu = make_srv18(tex, 0, 0, kMips);
        if (!draw_gradient_then_ui(gpu, "Path 1b", 0) || !verify_rt_for_mip("Path 1b", 0, true, false)) {
            std::fprintf(stderr, "FAIL Path 1b: CopyTextureRegion destination was overwritten from stale title_memory\n");
            record_fail(29);
        }
        invoke(up, 2);
        invoke(tex, 2);
        VirtualFree(backing, 0, MEM_RELEASE);
    }

    // =========================================================================
    // PATH 2: XGCreateTextureComputer (Alignment = 0, Layout = UNKNOWN)
    // =========================================================================
    auto run_xg_footprint_path = [&](const char* label, UINT desc_mips) -> bool {
        std::printf("%s\n", label);
        auto desc = mip_desc_3;
        desc.MipLevels = static_cast<UINT16>(desc_mips);
        void** computer = nullptr;
        if (FAILED(XGCreateTextureComputer(&desc, &computer, 0, 0)) || !computer) return false;
        auto* tbl = static_cast<void**>(computer[0]);
        using LayoutFn = HRESULT (WINAPI*)(void*, unsigned char*);
        using CopyIntoFn = HRESULT (WINAPI*)(void*, unsigned char*, UINT, UINT, const unsigned char*, UINT, UINT);
        std::array<unsigned char, 0x18 + 4 * 0x6c8 + 8> layout{};
        if (FAILED(reinterpret_cast<LayoutFn>(tbl[2])(computer, layout.data()))) return false;
        UINT64 total_bytes = 0; UINT32 reported_mips = 0;
        std::memcpy(&total_bytes, layout.data() + 0x00, 8);
        std::memcpy(&reported_mips, layout.data() + 0x10, 4);
        const UINT expected_mips = desc_mips ? desc_mips : 6u; // 32x32 full chain = 6 mips
        std::printf("  XG slot 2 reported_mips=%u (expected %u) total=0x%llx\n", reported_mips, expected_mips,
                    static_cast<unsigned long long>(total_bytes));
        if (reported_mips < kMips || total_bytes < mip_total) {
            std::fprintf(stderr, "  FAIL %s: XG reported %u mips instead of >=%u\n", label, reported_mips, kMips);
            reinterpret_cast<UINT64 (WINAPI*)(void*)>(tbl[1])(computer);
            return false;
        }
        auto* mem = static_cast<unsigned char*>(VirtualAlloc(nullptr, static_cast<SIZE_T>(total_bytes),
                                                             MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        std::memset(mem, 0, static_cast<std::size_t>(total_bytes));
        for (UINT m = 0; m < kMips; ++m) {
            const UINT w = kWidth >> m, h = kHeight >> m;
            std::vector<Rgba> lin(w * h);
            for (UINT y = 0; y < h; ++y)
                for (UINT x = 0; x < w; ++x) lin[y * w + x] = expected_texel(m, x, y);
            const HRESULT hr = reinterpret_cast<CopyIntoFn>(tbl[8])(computer, mem, 0, m,
                                                                    reinterpret_cast<const unsigned char*>(lin.data()), w * 4, 0);
            if (FAILED(hr)) {
                std::fprintf(stderr, "  FAIL %s: XG slot 8 subresource %u returned 0x%08lx\n",
                             label, m, static_cast<unsigned long>(hr));
                reinterpret_cast<UINT64 (WINAPI*)(void*)>(tbl[1])(computer);
                VirtualFree(mem, 0, MEM_RELEASE);
                return false;
            }
        }
        reinterpret_cast<UINT64 (WINAPI*)(void*)>(tbl[1])(computer);

        void* tex = nullptr;
        if (!ok(invoke(device, 133, arg(mem), arg(&desc), D3D12_RESOURCE_STATE_COMMON,
                       0, arg(&__uuidof(ID3D12Resource)), arg(&tex))) || !tex) {
            VirtualFree(mem, 0, MEM_RELEASE);
            return false;
        }
        for (UINT m = 0; m < kMips; ++m) {
            const Word gpu = make_srv18(tex, 0, m, kMips - m);
            if (!draw_gradient_then_ui(gpu, label, m) || !verify_rt_for_mip(label, m, true, false)) {
                invoke(tex, 2);
                VirtualFree(mem, 0, MEM_RELEASE);
                return false;
            }
        }
        std::vector<std::vector<Rgba>> rb_mips;
        if (!readback_texture_mips(tex, rb_mips)) { invoke(tex, 2); VirtualFree(mem, 0, MEM_RELEASE); return false; }
        for (UINT m = 0; m < kMips; ++m) {
            UINT bx = 0, by = 0; Rgba g{}, e{};
            if (find_first_texel_mismatch(rb_mips[m].data(), mip_fp[m].Footprint.RowPitch / 4, m, kWidth >> m, kHeight >> m, bx, by, g, e)) {
                std::fprintf(stderr, "  FAIL %s: GPU texel mismatch mip%u (%u,%u)\n", label, m, bx, by);
                invoke(tex, 2); VirtualFree(mem, 0, MEM_RELEASE); return false;
            }
        }
        invoke(tex, 2);
        VirtualFree(mem, 0, MEM_RELEASE);
        return true;
    };

    if (!run_xg_footprint_path("[Path 2a] XG footprint path (Alignment=0, MipLevels=3)", 3)) record_fail(30);
    if (!run_xg_footprint_path("[Path 2b] XG footprint path (Alignment=0, MipLevels=0 full chain)", 0)) record_fail(31);

    // =========================================================================
    // PATH 3: small_texture_4k compact path (Alignment = 0x1000, MipLevels = 3)
    // =========================================================================
    std::printf("[Path 3] small_texture_4k compact path (Alignment=0x1000, MipLevels=3)\n");
    {
        auto desc = mip_desc_3;
        desc.Alignment = 0x1000;
        void** computer = nullptr;
        if (FAILED(XGCreateTextureComputer(&desc, &computer, 0, 0)) || !computer) { record_fail(40); } else {
            auto* tbl = static_cast<void**>(computer[0]);
            using LayoutFn = HRESULT (WINAPI*)(void*, unsigned char*);
            using CopyIntoFn = HRESULT (WINAPI*)(void*, unsigned char*, UINT, UINT, const unsigned char*, UINT, UINT);
            std::array<unsigned char, 0x18 + 4 * 0x6c8 + 8> layout{};
            if (FAILED(reinterpret_cast<LayoutFn>(tbl[2])(computer, layout.data()))) { record_fail(41); }
            UINT64 total = 0, align = 0;
            std::memcpy(&total, layout.data() + 0x00, 8);
            std::memcpy(&align, layout.data() + 0x08, 8);
            std::printf("  small4k slot 2: total=0x%llx align=0x%llx\n",
                        static_cast<unsigned long long>(total), static_cast<unsigned long long>(align));
            if (total != 0x2000 || align != 0x1000) record_fail(42);
            auto* mem = static_cast<unsigned char*>(VirtualAlloc(nullptr, 0x3000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
            DWORD old = 0;
            if (!mem || !VirtualProtect(mem + 0x2000, 0x1000, PAGE_NOACCESS, &old)) record_fail(43);
            for (UINT m = 0; m < kMips; ++m) {
                const UINT w = kWidth >> m, h = kHeight >> m;
                std::vector<Rgba> lin(w * h);
                for (UINT y = 0; y < h; ++y)
                    for (UINT x = 0; x < w; ++x) lin[y * w + x] = expected_texel(m, x, y);
                if (FAILED(reinterpret_cast<CopyIntoFn>(tbl[8])(computer, mem, 0, m,
                                                                reinterpret_cast<const unsigned char*>(lin.data()), w * 4, 0))) {
                    record_fail(44);
                }
            }
            reinterpret_cast<UINT64 (WINAPI*)(void*)>(tbl[1])(computer);

            void* tex = nullptr;
            if (!ok(invoke(device, 133, arg(mem), arg(&desc), D3D12_RESOURCE_STATE_COMMON,
                           0, arg(&__uuidof(ID3D12Resource)), arg(&tex))) || !tex) {
                record_fail(45);
            } else {
                for (UINT m = 0; m < kMips; ++m) {
                    const Word gpu = make_srv18(tex, 0, m, kMips - m);
                    if (!draw_gradient_then_ui(gpu, "Path 3 (small4k)", m) || !verify_rt_for_mip("Path 3 (small4k)", m, true, false)) {
                        record_fail(46);
                    }
                }
                std::vector<std::vector<Rgba>> rb_mips;
                if (!readback_texture_mips(tex, rb_mips)) record_fail(47);
                else {
                    for (UINT m = 0; m < kMips; ++m) {
                        UINT bx = 0, by = 0; Rgba g{}, e{};
                        if (find_first_texel_mismatch(rb_mips[m].data(), mip_fp[m].Footprint.RowPitch / 4, m, kWidth >> m, kHeight >> m, bx, by, g, e)) {
                            std::fprintf(stderr, "FAIL Path 3: GPU texel mismatch mip%u (%u,%u)\n", m, bx, by);
                            record_fail(48);
                        }
                    }
                }
                invoke(tex, 2);
            }
            VirtualFree(mem, 0, MEM_RELEASE);
        }
    }

    // =========================================================================
    // PATH 4: Console-tiled gfx10::Layout via Device::slot70 (0x105 & 0x10a)
    // =========================================================================
    for (UINT layout_mode : {0x105u, 0x10au}) {
        std::printf("[Path 4] Console-tiled gfx10 via Device::slot70 (layout=0x%x, 3 mips)\n", layout_mode);
        Gfx10Tiler tiler;
        tiler.init(kWidth, kHeight, kMips, layout_mode);
        std::vector<unsigned char> tiled(tiler.total_bytes, 0);
        for (UINT m = 0; m < kMips; ++m) {
            const UINT w = kWidth >> m, h = kHeight >> m;
            for (UINT y = 0; y < h; ++y)
                for (UINT x = 0; x < w; ++x) {
                    const Rgba t = expected_texel(m, x, y);
                    std::memcpy(tiled.data() + tiler.addr(m, x, y), &t, 4);
                }
        }
        auto desc = mip_desc_3;
        desc.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(layout_mode);
        MemoryViewAt mv{};
        mv.view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        mv.view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        mv.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        mv.address = reinterpret_cast<Word>(tiled.data());
        for (UINT m = 0; m < kMips; ++m) {
            mv.view.Texture2D.MostDetailedMip = m;
            mv.view.Texture2D.MipLevels = kMips - m;
            invoke(device, 70, arg(&desc), arg(&mv), srv_cpu0);
            const char* lbl = layout_mode == 0x105 ? "Path 4 (slot70 0x105)" : "Path 4 (slot70 0x10a)";
            if (!draw_gradient_then_ui(srv_gpu0, lbl, m) || !verify_rt_for_mip(lbl, m, true, false)) {
                record_fail(50);
            }
        }
    }

    // =========================================================================
    // NEGATIVE CONTROL 1: Intentionally wrong row pitch (2x stride = 512 vs 256)
    // =========================================================================
    std::printf("[Neg Ctrl 1] Linear upload with intentionally wrong row pitch (512 stride read at 256 RowPitch)\n");
    {
        // Author mip 0 with stride 512 (rows at 0, 512, 1024...), leaving bytes [256..511] black (0),
        // then upload at desktop RowPitch 256 -> even rows get nonzero source rows, odd rows get zero gap!
        constexpr UINT wrong_stride = 512;
        const UINT upload_bytes = (kHeight * 2) * 256;
        std::vector<unsigned char> wrong_lin(kHeight * wrong_stride, 0);
        for (UINT y = 0; y < kHeight; ++y) {
            for (UINT x = 0; x < kWidth; ++x) {
                const Rgba t = expected_texel(0, x, y);
                std::memcpy(wrong_lin.data() + y * wrong_stride + x * 4, &t, 4);
            }
        }
        auto desc1 = mip_desc_3;
        desc1.MipLevels = 1;
        void* tex = make_committed(device, D3D12_HEAP_TYPE_DEFAULT, desc1, D3D12_RESOURCE_STATE_COMMON);
        void* up = make_committed(device, D3D12_HEAP_TYPE_UPLOAD, buffer_desc(upload_bytes), D3D12_RESOURCE_STATE_GENERIC_READ);
        void* mapped = nullptr;
        if (!tex || !up || !ok(invoke(up, 8, 0, 0, arg(&mapped))) || !mapped) { record_fail(60); } else {
            std::memcpy(mapped, wrong_lin.data(), kHeight * 256); // uploader thinks height=32 * pitch=256
            invoke(up, 9, 0, 0);

            if (!ok(invoke(alloc, 8)) || !ok(invoke(list, 139, arg(alloc), 0))) record_fail(61);
            D3D12_TEXTURE_COPY_LOCATION to{static_cast<ID3D12Resource*>(tex), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            D3D12_TEXTURE_COPY_LOCATION from{static_cast<ID3D12Resource*>(up), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            from.PlacedFootprint = {0, {DXGI_FORMAT_R8G8B8A8_UNORM, kWidth, kHeight, 1, 256}};
            invoke(list, 16, arg(&to), 0, 0, 0, arg(&from), 0);
            if (!ok(invoke(list, 9))) record_fail(62);
            void* lists[] = {list};
            invoke(queue, 10, 1, arg(lists));
            if (!ok(invoke(queue, 14, arg(fence), ++fence_val)) || !wait_fence(fence, fence_val)) record_fail(63);

            void* rb = make_committed(device, D3D12_HEAP_TYPE_READBACK, buffer_desc(mip_fp[0].Footprint.RowPitch * kHeight), D3D12_RESOURCE_STATE_COPY_DEST);
            void* rb_map = nullptr;
            if (!rb || !ok(invoke(alloc, 8)) || !ok(invoke(list, 139, arg(alloc), 0))) record_fail(64);
            D3D12_TEXTURE_COPY_LOCATION rb_dst{static_cast<ID3D12Resource*>(rb), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            rb_dst.PlacedFootprint = mip_fp[0];
            D3D12_TEXTURE_COPY_LOCATION rb_src{static_cast<ID3D12Resource*>(tex), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            invoke(list, 16, arg(&rb_dst), 0, 0, 0, arg(&rb_src), 0);
            if (!ok(invoke(list, 9))) record_fail(64);
            invoke(queue, 10, 1, arg(lists));
            if (!ok(invoke(queue, 14, arg(fence), ++fence_val)) || !wait_fence(fence, fence_val) ||
                !ok(invoke(rb, 8, 0, 0, arg(&rb_map))) || !rb_map) record_fail(64);
            std::vector<Rgba> gpu_rows((mip_fp[0].Footprint.RowPitch / 4) * kHeight);
            std::memcpy(gpu_rows.data(), rb_map, gpu_rows.size() * 4);
            invoke(rb, 9, 0, 0);
            invoke(rb, 2);
            print_rows_summary("neg1 gpu texels", gpu_rows.data(), mip_fp[0].Footprint.RowPitch / 4, kWidth, kHeight);
            UINT bx = 0, by = 0; Rgba g{}, e{};
            if (!find_first_texel_mismatch(gpu_rows.data(), mip_fp[0].Footprint.RowPitch / 4, 0, kWidth, kHeight, bx, by, g, e) ||
                by != 1 || bx != 0 || !g.is_black()) {
                std::fprintf(stderr, "FAIL Neg Ctrl 1: expected first GPU mismatch at (0,1)=black, got (%u,%u)\n", bx, by);
                record_fail(65);
            }
            std::printf("  Neg Ctrl 1: first GPU texel mismatch at (x=%u, y=%u, byte_offset=%u) got=(%u,%u,%u,%u) expected=(%u,%u,%u,%u)\n",
                        bx, by, by * mip_fp[0].Footprint.RowPitch + bx * 4, g.r, g.g, g.b, g.a, e.r, e.g, e.b, e.a);

            const Word gpu = make_srv18(tex, 0, 0, 1);
            if (!draw_gradient_then_ui(gpu, "Neg Ctrl 1", 0) || !verify_rt_for_mip("Neg Ctrl 1", 0, false, true)) {
                std::fprintf(stderr, "FAIL Neg Ctrl 1: did not reproduce alternating nonzero/black horizontal bands\n");
                record_fail(66);
            }
            invoke(up, 2);
            invoke(tex, 2);
        }
    }

    // =========================================================================
    // NEGATIVE CONTROL 2: Compact Alignment=0x1000 placed texture filled with desktop 256B row pitch
    // =========================================================================
    std::printf("[Neg Ctrl 2] Compact small_texture_4k (Alignment=0x1000) backed by 256B footprint rows (128B row mismatch)\n");
    {
        auto desc = mip_desc_3;
        desc.MipLevels = 1;
        desc.Alignment = 0x1000;
        const UINT64 alloc_bytes = 0x2000;
        auto* backing = static_cast<unsigned char*>(VirtualAlloc(nullptr, static_cast<SIZE_T>(alloc_bytes),
                                                               MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        std::memset(backing, 0, static_cast<std::size_t>(alloc_bytes));
        // Title placed 32x32 RGBA8 rows 256 bytes apart (128 bytes texels + 128 bytes zero padding) into Alignment=0x1000 memory;
        // runtime small_texture_4k reads compact 128-byte rows -> odd rows hit zero padding [128..255] of row y/2!
        for (UINT y = 0; y < kHeight / 2; ++y) {
            for (UINT x = 0; x < kWidth; ++x) {
                const Rgba t = expected_texel(0, x, y);
                std::memcpy(backing + y * 256 + x * 4, &t, 4);
            }
        }
        void* tex = nullptr;
        if (!ok(invoke(device, 133, arg(backing), arg(&desc), D3D12_RESOURCE_STATE_COMMON,
                       0, arg(&__uuidof(ID3D12Resource)), arg(&tex))) || !tex) {
            record_fail(70);
        } else {
            const Word gpu = make_srv18(tex, 0, 0, 1);
            if (!draw_gradient_then_ui(gpu, "Neg Ctrl 2", 0) || !verify_rt_for_mip("Neg Ctrl 2", 0, false, true)) {
                std::fprintf(stderr, "FAIL Neg Ctrl 2: Alignment=0x1000 pitch mismatch did not reproduce alternating bands\n");
                record_fail(71);
            } else {
                // Also read back GPU texels after upload to record the first mismatching byte/texel on GPU.
                void* rb = make_committed(device, D3D12_HEAP_TYPE_READBACK, buffer_desc(mip_fp[0].Footprint.RowPitch * kHeight),
                                          D3D12_RESOURCE_STATE_COPY_DEST);
                void* rb_map = nullptr;
                if (rb && ok(invoke(alloc, 8)) && ok(invoke(list, 139, arg(alloc), 0))) {
                    D3D12_TEXTURE_COPY_LOCATION rb_dst{static_cast<ID3D12Resource*>(rb), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
                    rb_dst.PlacedFootprint = mip_fp[0];
                    D3D12_TEXTURE_COPY_LOCATION rb_src{static_cast<ID3D12Resource*>(tex), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
                    invoke(list, 16, arg(&rb_dst), 0, 0, 0, arg(&rb_src), 0);
                    void* lists[] = {list};
                    if (ok(invoke(list, 9)) && (invoke(queue, 10, 1, arg(lists)), ok(invoke(queue, 14, arg(fence), ++fence_val))) &&
                        wait_fence(fence, fence_val) && ok(invoke(rb, 8, 0, 0, arg(&rb_map))) && rb_map) {
                        std::vector<Rgba> gpu_rows((mip_fp[0].Footprint.RowPitch / 4) * kHeight);
                        std::memcpy(gpu_rows.data(), rb_map, gpu_rows.size() * 4);
                        invoke(rb, 9, 0, 0);
                        print_rows_summary("neg2 gpu texels", gpu_rows.data(), mip_fp[0].Footprint.RowPitch / 4, kWidth, kHeight);
                        UINT bx = 0, by = 0; Rgba g{}, e{};
                        if (find_first_texel_mismatch(gpu_rows.data(), mip_fp[0].Footprint.RowPitch / 4, 0, kWidth, kHeight, bx, by, g, e)) {
                            std::printf("  Neg Ctrl 2: first GPU texel mismatch at (x=%u, y=%u, source_offset=%u -> gpu_offset=%u) "
                                        "got=(%u,%u,%u,%u) expected=(%u,%u,%u,%u)\n",
                                        bx, by, by * 128 + bx * 4, by * mip_fp[0].Footprint.RowPitch + bx * 4,
                                        g.r, g.g, g.b, g.a, e.r, e.g, e.b, e.a);
                        }
                    }
                    invoke(rb, 2);
                }
            }
            invoke(tex, 2);
        }
        VirtualFree(backing, 0, MEM_RELEASE);
    }

    invoke(fence, 2);
    invoke(list, 2);
    invoke(alloc, 2);
    invoke(queue, 2);
    invoke(srv_heap, 2);
    invoke(rtv_heap, 2);
    invoke(rt_readback, 2);
    invoke(rt, 2);
    invoke(tex_pso, 2);
    invoke(grad_pso, 2);
    invoke(guest_root, 2);
    invoke(device, 2);
    cleanup_cache();
    if (!all_ok) {
        std::printf("startup_ui_texture_bands_check: FAIL (%d)\n", first_fail_rc);
        return first_fail_rc;
    }
    std::printf("startup_ui_texture_bands_check: PASS\n");
    return 0;
}
