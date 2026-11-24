// Desktop pipeline-library cache for translated graphics PSOs
// (src/startup_stubs/graphics_pipeline_cache.h). WARP builds a trivial VS/PS,
// stores the pipeline library, reloads it through a fresh cache object and
// proves the warm pipeline can be recorded. Corrupt and disabled controls must
// still return a usable pipeline through CreateGraphicsPipelineState.

#include "xmem_test_device.h"
#include "../src/startup_stubs/graphics_pipeline_cache.h"
#include "desktop_gpu.h"

#include <d3dcompiler.h>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

static_assert(offsetof(D3D12_RENDER_TARGET_BLEND_DESC, RenderTargetWriteMask) + sizeof(UINT8) <
                  sizeof(D3D12_RENDER_TARGET_BLEND_DESC),
              "the blend-key control expects ABI padding after RenderTargetWriteMask");
static_assert(offsetof(D3D12_DEPTH_STENCIL_DESC, StencilWriteMask) + sizeof(UINT8) <
                  offsetof(D3D12_DEPTH_STENCIL_DESC, FrontFace),
              "the depth-key control expects ABI padding between the masks and FrontFace");

const char* const vertex_source = R"(
float4 main(uint id : SV_VertexID) : SV_POSITION
{
    return float4(id == 1 ? 3.0 : -1.0, id == 2 ? 3.0 : -1.0, 0.0, 1.0);
}
)";

const char* const pixel_source = R"(
float4 main() : SV_Target
{
    return float4(1.0, 0.5, 0.25, 1.0);
}
)";

bool compile(const char* source, const char* target, ID3DBlob** blob)
{
    ComPtr<ID3DBlob> errors;
    const HRESULT hr = D3DCompile(source, std::strlen(source), nullptr, nullptr, nullptr, "main", target,
                                  D3DCOMPILE_OPTIMIZATION_LEVEL1, 0, blob, &errors);
    if (FAILED(hr)) {
        std::printf("D3DCompile(%s) failed: 0x%08lx %s\n", target, static_cast<unsigned long>(hr),
                    errors ? static_cast<const char*>(errors->GetBufferPointer()) : "");
        return false;
    }
    return true;
}

bool make_description(ID3D12Device* device, ComPtr<ID3DBlob>& vs, ComPtr<ID3DBlob>& ps,
                      ComPtr<ID3D12RootSignature>& root, D3D12_GRAPHICS_PIPELINE_STATE_DESC* desc)
{
    if (!compile(vertex_source, "vs_5_0", vs.GetAddressOf()) ||
        !compile(pixel_source, "ps_5_0", ps.GetAddressOf()))
        return false;
    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    ComPtr<ID3DBlob> signature;
    if (FAILED(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, nullptr)) ||
        FAILED(device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                           IID_PPV_ARGS(&root))))
        return false;
    *desc = {};
    desc->pRootSignature = root.Get();
    desc->VS = {vs->GetBufferPointer(), vs->GetBufferSize()};
    desc->PS = {ps->GetBufferPointer(), ps->GetBufferSize()};
    for (auto& target : desc->BlendState.RenderTarget)
        target = {FALSE, FALSE, D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD, D3D12_BLEND_ONE,
                  D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD, D3D12_LOGIC_OP_NOOP, D3D12_COLOR_WRITE_ENABLE_ALL};
    desc->SampleMask = ~0u;
    desc->RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc->RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc->RasterizerState.DepthClipEnable = TRUE;
    desc->PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc->NumRenderTargets = 1;
    desc->RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc->SampleDesc.Count = 1;
    (void)device;
    return true;
}

bool usable(ID3D12Device* device, ID3D12PipelineState* pipeline)
{
    if (!pipeline) return false;
    ComPtr<ID3D12CommandAllocator> allocator;
    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))))
        return false;
    ComPtr<ID3D12GraphicsCommandList> list;
    if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                         IID_PPV_ARGS(&list))))
        return false;
    list->SetPipelineState(pipeline);
    return SUCCEEDED(list->Close());
}

bool render_color(ID3D12Device* device, ID3D12PipelineState* pipeline, std::uint32_t* out)
{
    constexpr UINT width = 4, height = 4;
    D3D12_RESOURCE_DESC target_desc{};
    target_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    target_desc.Width = width;
    target_desc.Height = height;
    target_desc.DepthOrArraySize = 1;
    target_desc.MipLevels = 1;
    target_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    target_desc.SampleDesc.Count = 1;
    target_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    const D3D12_CLEAR_VALUE clear{DXGI_FORMAT_R8G8B8A8_UNORM, {0.0f, 0.0f, 0.0f, 1.0f}};
    const auto target = resource(device, target_desc, D3D12_HEAP_TYPE_DEFAULT,
                                 D3D12_RESOURCE_STATE_RENDER_TARGET, &clear);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 readback_size = 0;
    device->GetCopyableFootprints(&target_desc, 0, 1, 0, &footprint, nullptr, nullptr, &readback_size);
    const auto readback = make_buffer(device, readback_size, D3D12_HEAP_TYPE_READBACK,
                                      D3D12_RESOURCE_STATE_COPY_DEST);

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
    ComPtr<ID3D12DescriptorHeap> rtvs;
    if (FAILED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtvs)))) return false;
    const auto rtv = rtvs->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.Get(), nullptr, rtv);

    ComputeCommands commands(device, pipeline, D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto* list = commands.list.Get();
    list->ClearRenderTargetView(rtv, clear.Color, 0, nullptr);
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    const D3D12_VIEWPORT viewport{0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
    const D3D12_RECT scissor{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->DrawInstanced(3, 1, 0, 0);
    transition(list, target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION dst{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    dst.PlacedFootprint = footprint;
    D3D12_TEXTURE_COPY_LOCATION src{target.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    commands.submit_and_wait(device);

    void* mapped = nullptr;
    const D3D12_RANGE read{0, static_cast<SIZE_T>(readback_size)};
    const D3D12_RANGE none{0, 0};
    if (FAILED(readback->Map(0, &read, &mapped))) return false;
    std::memcpy(out, static_cast<const std::uint8_t*>(mapped) + footprint.Offset, 4 * sizeof(std::uint32_t));
    readback->Unmap(0, &none);
    return true;
}

bool timed_batch(ID3D12Device* device, const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc, bool use_cache,
                 int count, double* total_ms, double* flush_ms)
{
    LARGE_INTEGER frequency, start, stored, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    {
        x2w_startup::GraphicsPipelineCache cache(device, use_cache);
        for (int i = 0; i < count; ++i) {
            ComPtr<ID3D12PipelineState> pso;
            if (FAILED(cache.GetOrCreateGraphicsPipeline(desc, 0x2000ull + static_cast<std::uint64_t>(i),
                                                         IID_PPV_ARGS(&pso))) ||
                !pso)
                return false;
        }
        QueryPerformanceCounter(&stored);
        cache.Flush();
        QueryPerformanceCounter(&end);
    }
    *total_ms = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
    if (flush_ms) *flush_ms = (end.QuadPart - stored.QuadPart) * 1000.0 / frequency.QuadPart;
    return true;
}

bool exists(const wchar_t* path)
{
    return GetFileAttributesW(path) != INVALID_FILE_ATTRIBUTES;
}

bool corrupt(const wchar_t* path)
{
    HANDLE file = CreateFileW(path, GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    const char bytes[] = "truncated-cache";
    DWORD written = 0;
    const bool ok = WriteFile(file, bytes, sizeof(bytes) - 1, &written, nullptr) && written == sizeof(bytes) - 1 &&
                    SetEndOfFile(file);
    CloseHandle(file);
    return ok;
}

bool hash_padding_ignored()
{
    using namespace x2w_startup::graphics_pipeline_cache_detail;

    D3D12_BLEND_DESC blend_a{};
    blend_a.AlphaToCoverageEnable = TRUE;
    blend_a.IndependentBlendEnable = FALSE;
    for (auto& target : blend_a.RenderTarget) {
        target = {TRUE, FALSE, D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_OP_ADD,
                  D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD, D3D12_LOGIC_OP_NOOP,
                  D3D12_COLOR_WRITE_ENABLE_ALL};
    }
    D3D12_BLEND_DESC blend_b = blend_a;
    for (auto& target : blend_b.RenderTarget) {
        auto* bytes = reinterpret_cast<unsigned char*>(&target);
        for (std::size_t i = offsetof(D3D12_RENDER_TARGET_BLEND_DESC, RenderTargetWriteMask) +
                             sizeof(target.RenderTargetWriteMask);
             i < sizeof(target); ++i)
            bytes[i] = 0xa5;
    }
    const std::uint64_t seed = 0x123456789abcdef0ull;
    if (hash_blend_key(blend_a, seed) != hash_blend_key(blend_b, seed)) return false;
    blend_b.RenderTarget[0].SrcBlend = D3D12_BLEND_ONE;
    if (hash_blend_key(blend_a, seed) == hash_blend_key(blend_b, seed)) return false;

    D3D12_DEPTH_STENCIL_DESC depth_a{};
    depth_a.DepthEnable = TRUE;
    depth_a.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    depth_a.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    depth_a.StencilEnable = TRUE;
    depth_a.StencilReadMask = 0x3f;
    depth_a.StencilWriteMask = 0x7f;
    depth_a.FrontFace = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_INCR, D3D12_STENCIL_OP_KEEP,
                         D3D12_COMPARISON_FUNC_ALWAYS};
    depth_a.BackFace = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_DECR, D3D12_STENCIL_OP_KEEP,
                        D3D12_COMPARISON_FUNC_NEVER};
    D3D12_DEPTH_STENCIL_DESC depth_b = depth_a;
    auto* depth_bytes = reinterpret_cast<unsigned char*>(&depth_b);
    for (std::size_t i = offsetof(D3D12_DEPTH_STENCIL_DESC, StencilWriteMask) + sizeof(depth_b.StencilWriteMask);
         i < offsetof(D3D12_DEPTH_STENCIL_DESC, FrontFace); ++i)
        depth_bytes[i] = 0x5a;
    if (hash_depth_stencil_key(depth_a, seed) != hash_depth_stencil_key(depth_b, seed)) return false;
    depth_b.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_EQUAL;
    return hash_depth_stencil_key(depth_a, seed) != hash_depth_stencil_key(depth_b, seed);
}

template <typename Change>
bool blend_field_changes(const D3D12_BLEND_DESC& base, Change change)
{
    static constexpr std::uint64_t seed = 0x123456789abcdef0ull;
    D3D12_BLEND_DESC modified = base;
    change(modified);
    return x2w_startup::graphics_pipeline_cache_detail::hash_blend_key(base, seed) !=
           x2w_startup::graphics_pipeline_cache_detail::hash_blend_key(modified, seed);
}

template <typename Change>
bool depth_field_changes(const D3D12_DEPTH_STENCIL_DESC& base, Change change)
{
    static constexpr std::uint64_t seed = 0x123456789abcdef0ull;
    D3D12_DEPTH_STENCIL_DESC modified = base;
    change(modified);
    return x2w_startup::graphics_pipeline_cache_detail::hash_depth_stencil_key(base, seed) !=
           x2w_startup::graphics_pipeline_cache_detail::hash_depth_stencil_key(modified, seed);
}

bool hash_fields_covered()
{
    D3D12_BLEND_DESC blend{};
    blend.AlphaToCoverageEnable = TRUE;
    blend.IndependentBlendEnable = FALSE;
    for (auto& target : blend.RenderTarget) {
        target = {TRUE, FALSE, D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_OP_ADD,
                  D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD, D3D12_LOGIC_OP_NOOP,
                  D3D12_COLOR_WRITE_ENABLE_ALL};
    }

    bool ok = true;
    if (!blend_field_changes(blend, [](auto& b) { b.AlphaToCoverageEnable = FALSE; })) {
        std::puts("FAIL: blend AlphaToCoverageEnable is not in the key");
        ok = false;
    }
    if (!blend_field_changes(blend, [](auto& b) { b.IndependentBlendEnable = TRUE; })) {
        std::puts("FAIL: blend IndependentBlendEnable is not in the key");
        ok = false;
    }
    for (int target = 0; target < 8; target += 7) {
        auto field = [&](const char* name, auto change) {
            if (!blend_field_changes(blend, change)) {
                std::printf("FAIL: blend target %d %s is not in the key\n", target, name);
                ok = false;
            }
        };
        field("BlendEnable", [target](auto& b) { b.RenderTarget[target].BlendEnable = FALSE; });
        field("LogicOpEnable", [target](auto& b) { b.RenderTarget[target].LogicOpEnable = TRUE; });
        field("SrcBlend", [target](auto& b) { b.RenderTarget[target].SrcBlend = D3D12_BLEND_ONE; });
        field("DestBlend", [target](auto& b) { b.RenderTarget[target].DestBlend = D3D12_BLEND_ONE; });
        field("BlendOp", [target](auto& b) { b.RenderTarget[target].BlendOp = D3D12_BLEND_OP_SUBTRACT; });
        field("SrcBlendAlpha", [target](auto& b) { b.RenderTarget[target].SrcBlendAlpha = D3D12_BLEND_ZERO; });
        field("DestBlendAlpha", [target](auto& b) { b.RenderTarget[target].DestBlendAlpha = D3D12_BLEND_ONE; });
        field("BlendOpAlpha", [target](auto& b) { b.RenderTarget[target].BlendOpAlpha = D3D12_BLEND_OP_MAX; });
        field("LogicOp", [target](auto& b) { b.RenderTarget[target].LogicOp = D3D12_LOGIC_OP_CLEAR; });
        field("RenderTargetWriteMask", [target](auto& b) {
            b.RenderTarget[target].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_RED;
        });
    }

    D3D12_DEPTH_STENCIL_DESC depth{};
    depth.DepthEnable = TRUE;
    depth.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    depth.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    depth.StencilEnable = TRUE;
    depth.StencilReadMask = 0x3f;
    depth.StencilWriteMask = 0x7f;
    depth.FrontFace = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_INCR, D3D12_STENCIL_OP_KEEP,
                       D3D12_COMPARISON_FUNC_ALWAYS};
    depth.BackFace = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_DECR, D3D12_STENCIL_OP_KEEP,
                      D3D12_COMPARISON_FUNC_NEVER};

    auto depth_field = [&](const char* name, auto change) {
        if (!depth_field_changes(depth, change)) {
            std::printf("FAIL: depth %s is not in the key\n", name);
            ok = false;
        }
    };
    depth_field("DepthEnable", [](auto& d) { d.DepthEnable = FALSE; });
    depth_field("DepthWriteMask", [](auto& d) { d.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO; });
    depth_field("DepthFunc", [](auto& d) { d.DepthFunc = D3D12_COMPARISON_FUNC_GREATER; });
    depth_field("StencilEnable", [](auto& d) { d.StencilEnable = FALSE; });
    depth_field("StencilReadMask", [](auto& d) { d.StencilReadMask = 0x1f; });
    depth_field("StencilWriteMask", [](auto& d) { d.StencilWriteMask = 0x3f; });
    depth_field("FrontFace.StencilFailOp", [](auto& d) { d.FrontFace.StencilFailOp = D3D12_STENCIL_OP_ZERO; });
    depth_field("FrontFace.StencilDepthFailOp",
                [](auto& d) { d.FrontFace.StencilDepthFailOp = D3D12_STENCIL_OP_ZERO; });
    depth_field("FrontFace.StencilPassOp", [](auto& d) { d.FrontFace.StencilPassOp = D3D12_STENCIL_OP_ZERO; });
    depth_field("FrontFace.StencilFunc", [](auto& d) { d.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_EQUAL; });
    depth_field("BackFace.StencilFailOp", [](auto& d) { d.BackFace.StencilFailOp = D3D12_STENCIL_OP_ZERO; });
    depth_field("BackFace.StencilDepthFailOp",
                [](auto& d) { d.BackFace.StencilDepthFailOp = D3D12_STENCIL_OP_ZERO; });
    depth_field("BackFace.StencilPassOp", [](auto& d) { d.BackFace.StencilPassOp = D3D12_STENCIL_OP_ZERO; });
    depth_field("BackFace.StencilFunc", [](auto& d) { d.BackFace.StencilFunc = D3D12_COMPARISON_FUNC_EQUAL; });
    return ok;
}

bool code_key_changes_pipeline_name()
{
    using namespace x2w_startup::graphics_pipeline_cache_detail;
    const DXGI_FORMAT formats[] = {DXGI_FORMAT_R8G8B8A8_UNORM};
    const std::uint64_t translation = 0x0123456789abcdefull;
    const std::uint64_t code_a = translated_code_key(translation, "vs-a", 4, "ps-a", 4);
    const std::uint64_t vs_changed = translated_code_key(translation, "vs-b", 4, "ps-a", 4);
    const std::uint64_t ps_changed = translated_code_key(translation, "vs-a", 4, "ps-b", 4);
    const std::uint64_t key_a = pipeline_key(translation, code_a, formats, 1, DXGI_FORMAT_UNKNOWN,
                                             false, nullptr, false, nullptr);
    const std::uint64_t key_vs = pipeline_key(translation, vs_changed, formats, 1,
                                              DXGI_FORMAT_UNKNOWN, false, nullptr, false, nullptr);
    const std::uint64_t key_ps = pipeline_key(translation, ps_changed, formats, 1,
                                              DXGI_FORMAT_UNKNOWN, false, nullptr, false, nullptr);
    const std::uint64_t key_old_wiring = pipeline_key(translation, 0, formats, 1,
                                                      DXGI_FORMAT_UNKNOWN, false, nullptr, false,
                                                      nullptr);
    const std::uint64_t key_a_again = pipeline_key(translation, code_a, formats, 1,
                                                   DXGI_FORMAT_UNKNOWN, false, nullptr, false,
                                                   nullptr);
    if (code_a == vs_changed || code_a == ps_changed) {
        std::puts("FAIL: translated VS/PS bytes do not change the code key");
        return false;
    }
    if (key_a == key_vs || key_a == key_ps) {
        std::puts("FAIL: translated VS/PS bytes do not change the pipeline name");
        return false;
    }
    if (key_a == key_old_wiring) {
        std::puts("FAIL: dropping code_key still produces the old pipeline name");
        return false;
    }
    if (key_a != key_a_again) {
        std::puts("FAIL: the same code key does not produce a stable pipeline name");
        return false;
    }
    return true;
}

void remove_cache(const wchar_t* directory)
{
    wchar_t pattern[MAX_PATH + 8]{};
    swprintf(pattern, sizeof(pattern) / sizeof(pattern[0]), L"%s\\pso_*", directory);
    WIN32_FIND_DATAW data{};
    HANDLE find = FindFirstFileW(pattern, &data);
    if (find != INVALID_HANDLE_VALUE) {
        do {
            wchar_t path[MAX_PATH * 2]{};
            swprintf(path, sizeof(path) / sizeof(path[0]), L"%s\\%s", directory, data.cFileName);
            DeleteFileW(path);
        } while (FindNextFileW(find, &data));
        FindClose(find);
    }
    RemoveDirectoryW(directory);
}

}  // namespace

int main()
{
    bool unmatched = false;
    ComPtr<ID3D12Device> device = create_device(true, unmatched);
    if (!device) {
        std::puts("SKIP: no WARP D3D12 device");
        return 77;
    }
    const std::uint64_t identity =
        x2w_startup::graphics_pipeline_cache_detail::adapter_identity(device.Get());
    const std::uint64_t seed =
        x2w_startup::graphics_pipeline_cache_detail::fnv_bytes("x2w-pso-cache-v2", 16);
    std::printf("adapter identity %016llx\n", static_cast<unsigned long long>(identity));
    if (identity == seed) {
        std::puts("FAIL: adapter identity is still the constant seed");
        return 1;
    }

    wchar_t temp[MAX_PATH]{};
    if (!GetTempPathW(MAX_PATH, temp)) return 1;
    wchar_t directory[MAX_PATH]{};
    swprintf(directory, sizeof(directory) / sizeof(directory[0]), L"%sx2w_pso_cache_%lu", temp,
             GetCurrentProcessId());
    CreateDirectoryW(directory, nullptr);
    if (!SetEnvironmentVariableW(L"X2W_SHADER_CACHE", directory)) return 1;

    ComPtr<ID3DBlob> vs, ps;
    ComPtr<ID3D12RootSignature> root;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    if (!make_description(device.Get(), vs, ps, root, &desc)) return 1;
    const std::uint64_t key = 0x0123456789abcdefull;

    bool ok = true;
    if (!hash_padding_ignored()) {
        std::puts("FAIL: blend/depth key includes ABI padding or ignores a semantic field");
        ok = false;
    }
    if (!hash_fields_covered()) {
        std::puts("FAIL: at least one blend/depth field is missing from the key");
        ok = false;
    }
    if (!code_key_changes_pipeline_name()) {
        std::puts("FAIL: the translated code bytes are not part of the pipeline name");
        ok = false;
    }
    std::uint32_t cold_pixel[4]{};
    bool have_cold_pixel = false;
    {
        x2w_startup::GraphicsPipelineCache cache(device.Get());
        if (!cache.configured()) {
            std::puts("FAIL: X2W_SHADER_CACHE was not visible to the cache helper");
            ok = false;
        }
        if (!cache.enabled()) {
            std::puts("SKIP: ID3D12PipelineLibrary is unavailable; disabled fallback is tested below");
        } else {
            ComPtr<ID3D12PipelineState> cold;
            bool hit = true;
            const HRESULT hr = cache.GetOrCreateGraphicsPipeline(desc, key, IID_PPV_ARGS(&cold), &hit);
            if (FAILED(hr) || !cold || hit) {
                std::printf("FAIL: cold cache call hr=0x%08lx hit=%d pso=%p\n", static_cast<unsigned long>(hr),
                            hit ? 1 : 0, static_cast<void*>(cold.Get()));
                ok = false;
            }
            if (cold) {
                have_cold_pixel = render_color(device.Get(), cold.Get(), cold_pixel);
                if (!have_cold_pixel) {
                    std::puts("FAIL: cold pipeline did not render a pixel");
                    ok = false;
                }
            }
            cache.Flush();
            if (!exists(cache.path())) {
                std::puts("FAIL: cold cache call did not write a pso_ library blob");
                ok = false;
            }
        }
    }

    {
        x2w_startup::GraphicsPipelineCache warm(device.Get());
        if (warm.enabled()) {
            ComPtr<ID3D12PipelineState> pso;
            bool hit = false;
            const HRESULT hr = warm.GetOrCreateGraphicsPipeline(desc, key, IID_PPV_ARGS(&pso), &hit);
            if (FAILED(hr) || !pso || !hit || !usable(device.Get(), pso.Get())) {
                std::printf("FAIL: fresh-library warm hit hr=0x%08lx hit=%d pso=%p\n",
                            static_cast<unsigned long>(hr), hit ? 1 : 0, static_cast<void*>(pso.Get()));
                ok = false;
            }
            if (pso && have_cold_pixel) {
                std::uint32_t warm_pixel[4]{};
                if (!render_color(device.Get(), pso.Get(), warm_pixel) ||
                    std::memcmp(cold_pixel, warm_pixel, sizeof(cold_pixel)) != 0) {
                    std::puts("FAIL: warm pipeline rendered a different pixel");
                    ok = false;
                }
                constexpr std::uint32_t expected = 0xff4080ffu;
                for (std::uint32_t value : warm_pixel) {
                    if (value != expected) {
                        std::printf("FAIL: warm pixel %08x, expected %08x\n", value, expected);
                        ok = false;
                        break;
                    }
                }
            }
        }
    }

    {
        x2w_startup::GraphicsPipelineCache corrupt_cache(device.Get());
        if (corrupt_cache.enabled()) {
            if (!corrupt(corrupt_cache.path())) {
                std::puts("FAIL: could not truncate the cache blob");
                ok = false;
            }
            x2w_startup::GraphicsPipelineCache recovered(device.Get());
            ComPtr<ID3D12PipelineState> pso;
            bool hit = true;
            const HRESULT hr = recovered.GetOrCreateGraphicsPipeline(desc, key, IID_PPV_ARGS(&pso), &hit);
            if (FAILED(hr) || !pso || hit || !usable(device.Get(), pso.Get())) {
                std::printf("FAIL: corrupt-blob fallback hr=0x%08lx hit=%d pso=%p\n",
                            static_cast<unsigned long>(hr), hit ? 1 : 0, static_cast<void*>(pso.Get()));
                ok = false;
            }
        }
    }

    {
        x2w_startup::GraphicsPipelineCache disabled(device.Get(), false);
        ComPtr<ID3D12PipelineState> pso;
        bool hit = true;
        const HRESULT hr = disabled.GetOrCreateGraphicsPipeline(desc, 77, IID_PPV_ARGS(&pso), &hit);
        if (disabled.enabled() || FAILED(hr) || !pso || hit || !usable(device.Get(), pso.Get())) {
            std::printf("FAIL: unsupported-interface fallback enabled=%d hr=0x%08lx hit=%d pso=%p\n",
                        disabled.enabled() ? 1 : 0, static_cast<unsigned long>(hr), hit ? 1 : 0,
                        static_cast<void*>(pso.Get()));
            ok = false;
        }
    }

    {
        x2w_startup::GraphicsPipelineCache concurrent(device.Get());
        std::atomic<int> failures{0};
        std::atomic<int> hits{0};
        std::vector<ID3D12PipelineState*> pipelines(4, nullptr);
        std::vector<std::thread> threads;
        for (int i = 0; i < 4; ++i) {
            threads.emplace_back([&, i] {
                bool hit = false;
                const HRESULT hr = concurrent.GetOrCreateGraphicsPipeline(desc, 99, IID_PPV_ARGS(&pipelines[i]), &hit);
                if (FAILED(hr) || !pipelines[i]) ++failures;
                if (hit) ++hits;
            });
        }
        for (auto& thread : threads) thread.join();
        for (auto* pipeline : pipelines) {
            if (!usable(device.Get(), pipeline)) ++failures;
            if (pipeline) pipeline->Release();
        }
        if (failures || (concurrent.enabled() && hits == 0)) {
            std::printf("FAIL: concurrent creation failures=%d hits=%d enabled=%d\n", failures.load(), hits.load(),
                        concurrent.enabled() ? 1 : 0);
            ok = false;
        }
    }

    {
        remove_cache(directory);
        CreateDirectoryW(directory, nullptr);
        double uncached = 0, cached = 0, flush = 0, warm = 0;
        if (!timed_batch(device.Get(), desc, false, 8, &uncached, nullptr) ||
            !timed_batch(device.Get(), desc, true, 8, &cached, &flush) ||
            !timed_batch(device.Get(), desc, true, 8, &warm, nullptr)) {
            std::puts("FAIL: PSO batch timing failed");
            ok = false;
        } else {
            std::printf("PSO batch 8: uncached %.2f ms, cached cold %.2f ms (flush %.2f ms), warm %.2f ms\n",
                        uncached, cached, flush, warm);
        }
    }

    std::printf("startup_pso_cache %s (library %s)\n", ok ? "ok" : "FAILED",
                device ? "WARP" : "none");
    remove_cache(directory);
    return ok ? 0 : 1;
}
