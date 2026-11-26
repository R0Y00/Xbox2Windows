// The shader binding ABI switch (src/startup_stubs/gpu_abi.h, gpu_hooks.inl; PLAN.md WP0) through
// D3D12XboxCreateDevice on WARP.
//
//   startup_gpu_abi_off_check off [--mutate]    X2W_GPU_ABI unset (the test clears it)
//   startup_gpu_abi_off_check on                X2W_GPU_ABI=1 (the test sets it)
//
// The same call sequence in both modes: device slot 15 for the four desktop heap types, slot 14 (a CPU view heap),
// slot 18 (a null SRV into a title-memory heap), slot 57 (a zero description), slot 62 (a compute head with a synthetic container) and slot 86 (a stream with a mesh
// and a pixel stage). X2W_GPU_KEYLOG names a file in %TEMP% in both modes.
// - off: every answer is the old one (slot 15 = the desktop increments, 57 hands out a placeholder), no hook counter
//   (x2w_gpu_counter) moves, and no key file is written.
// - on: slot 15 answers 32 for views and 16 for samplers (RTV and DSV as desktop), 57 refuses with E_NOTIMPL and a null
//   out, 62 and 86 hand out placeholders, the counters show 2 pipelines, 3 keys, 1 refusal, 3 heaps, 1 view, 1 copy, the key file
//   holds the blake3 keys (first 16 hex digits) of the three containers with their stages.
// - both: a view at slot 3 of a title-memory heap sized by the slot 15 answer lands 3 increments in (title_stride), and
//   slot 24 copies it to slot 5 of a second such heap.
// --mutate sets the switch although the mode is "off" (the default flipped): the off checks must fail.
//
// Exit codes: 0 pass, 3 no WARP adapter, 4 no device, 10 + bits for failed checks.
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "startup_stubs/blake3/blake3.h"
#include "startup_stubs/gpu_abi.h"

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) long long x2w_gpu_counter(int which);

namespace {
using Word = std::uint64_t;
void* slot(void* object, int n) { return (*static_cast<void***>(object))[n]; }

std::string key_of(const void* bytes, std::size_t size)
{
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, bytes, size);
    std::uint8_t hash[8];
    blake3_hasher_finalize(&hasher, hash, sizeof(hash));
    char text[17];
    for (int i = 0; i < 8; ++i) std::snprintf(text + 2 * i, 3, "%02x", hash[i]);
    return text;
}

std::vector<unsigned char> container(unsigned char seed, std::size_t size)
{
    std::vector<unsigned char> bytes(size);
    std::memcpy(bytes.data(), "DXBC", 4);
    for (std::size_t i = 4; i < size; ++i) bytes[i] = static_cast<unsigned char>(seed + i * 7);
    return bytes;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) return 2;
    const bool on = std::strcmp(argv[1], "on") == 0;
    const bool mutate = argc > 2 && std::strcmp(argv[2], "--mutate") == 0;
    SetEnvironmentVariableA("X2W_GPU_ABI", on || mutate ? "1" : nullptr);
    char temp[MAX_PATH];
    GetTempPathA(MAX_PATH, temp);
    char keylog[MAX_PATH];
    std::snprintf(keylog, sizeof(keylog), "%sx2w_gpu_keys_%lu.txt", temp, GetCurrentProcessId());
    DeleteFileA(keylog);
    SetEnvironmentVariableA("X2W_GPU_KEYLOG", keylog);

    int failed = 0;
    const auto check = [&](bool ok, int bit, const char* what) {
        std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
        if (!ok) failed |= bit;
    };
    // The vendored hash: blake3("") = af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262.
    check(key_of("", 0) == "af1349b9f5f9a1a6", 1, "blake3 of the empty input");

    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter> adapter;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) return 3;
    unsigned char parameters[64]{};
    void* device = nullptr;
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters, __uuidof(ID3D12Device), &device))) return 4;
    ComPtr<ID3D12Device> desktop;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&desktop)))) return 4;

    // Slot 15.
    using Increment = UINT(WINAPI*)(void*, Word);
    for (UINT type = 0; type < D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES; ++type) {
        const UINT want_desktop = desktop->GetDescriptorHandleIncrementSize(static_cast<D3D12_DESCRIPTOR_HEAP_TYPE>(type));
        const UINT got = reinterpret_cast<Increment>(slot(device, 15))(device, type);
        UINT want = want_desktop;
        if (on && type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) want = 32;
        if (on && type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) want = 16;
        char what[96];
        std::snprintf(what, sizeof(what), "slot 15 type %u: %u (want %u, desktop %u)", type, got, want, want_desktop);
        check(got == want, 2, what);
    }

    // Slot 14: a CPU-only view heap, followed by the (absent) title memory extension.
    struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        std::uint64_t extra[2];
    } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 16, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0}, {0, 0}};
    using CreateHeap = HRESULT(WINAPI*)(void*, const void*, const GUID*, void**);
    void* heap = nullptr;
    const HRESULT heap_hr = reinterpret_cast<CreateHeap>(slot(device, 14))(device, &heap_desc, &__uuidof(ID3D12DescriptorHeap), &heap);
    check(SUCCEEDED(heap_hr) && heap, 4, "slot 14 view heap created");

    // A view heap over title memory sized by the slot 15 answer: a view at slot 3 must land at 3 increments (title_stride
    // follows slot 15), seen as the "X2WV" token X2W writes into title memory there.
    const UINT view_increment = reinterpret_cast<Increment>(slot(device, 15))(device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    std::vector<unsigned char> title_views(16 * std::size_t{view_increment});
    struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        std::uint64_t extra[2];
    } title_heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 16, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0},
                      {reinterpret_cast<std::uint64_t>(title_views.data()), title_views.size()}};
    void* title_heap = nullptr;
    const HRESULT title_heap_hr =
        reinterpret_cast<CreateHeap>(slot(device, 14))(device, &title_heap_desc, &__uuidof(ID3D12DescriptorHeap), &title_heap);
    check(SUCCEEDED(title_heap_hr) && title_heap, 4, "slot 14 view heap over title memory created");
    D3D12_SHADER_RESOURCE_VIEW_DESC null_srv{};
    null_srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    null_srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    null_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    null_srv.Texture2D.MipLevels = 1;
    using CreateSrv = void(WINAPI*)(void*, void*, const void*, Word);
    reinterpret_cast<CreateSrv>(slot(device, 18))(device, nullptr, &null_srv,
                                                  reinterpret_cast<Word>(title_views.data()) + 3 * Word{view_increment});
    std::uint32_t token = 0;
    std::memcpy(&token, title_views.data() + 3 * std::size_t{view_increment}, sizeof(token));
    char token_what[96];
    std::snprintf(token_what, sizeof(token_what), "view token at slot 3 of a %u-byte-stride title heap: 0x%08x", view_increment, token);
    check(token == 0x56573258u, 4, token_what);
    // CopyDescriptorsSimple (slot 24) of that slot into slot 5 of a second title-memory heap: the token moves with the
    // title stride (WARP: title 32 under ABI, desktop 64).
    std::vector<unsigned char> title_views2(16 * std::size_t{view_increment});
    auto title_heap_desc2 = title_heap_desc;
    title_heap_desc2.extra[0] = reinterpret_cast<std::uint64_t>(title_views2.data());
    void* title_heap2 = nullptr;
    reinterpret_cast<CreateHeap>(slot(device, 14))(device, &title_heap_desc2, &__uuidof(ID3D12DescriptorHeap), &title_heap2);
    using CopySimple = void(WINAPI*)(void*, UINT, Word, Word, D3D12_DESCRIPTOR_HEAP_TYPE);
    reinterpret_cast<CopySimple>(slot(device, 24))(device, 1, reinterpret_cast<Word>(title_views2.data()) + 5 * Word{view_increment},
                                                   reinterpret_cast<Word>(title_views.data()) + 3 * Word{view_increment},
                                                   D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    std::uint32_t copied = 0;
    std::memcpy(&copied, title_views2.data() + 5 * std::size_t{view_increment}, sizeof(copied));
    std::snprintf(token_what, sizeof(token_what), "copied view token at slot 5 of the second heap: 0x%08x", copied);
    check(title_heap2 && copied == 0x56573258u, 4, token_what);

    // Slot 57 with a zero description.
    using Create57 = HRESULT(WINAPI*)(void*, const void*, const GUID*, void**);
    std::vector<unsigned char> zero(0x300);
    void* refused = reinterpret_cast<void*>(1);
    const HRESULT hr57 = reinterpret_cast<Create57>(slot(device, 57))(device, zero.data(), &__uuidof(ID3D12PipelineState), &refused);
    if (on) check(hr57 == E_NOTIMPL && !refused, 8, "slot 57 refused with E_NOTIMPL, out null");
    else check(hr57 == S_OK && refused, 8, "slot 57 hands out a placeholder");

    // Slot 62: a compute head with a synthetic container and no root.
    const auto cs = container(1, 200);
    D3D12_COMPUTE_PIPELINE_STATE_DESC compute{};
    compute.CS = {cs.data(), cs.size()};
    using Create62 = HRESULT(WINAPI*)(void*, const void*, Word, Word, const GUID*, void**);
    void* pipeline62 = nullptr;
    const HRESULT hr62 = reinterpret_cast<Create62>(slot(device, 62))(device, &compute, 0, 0, &__uuidof(ID3D12PipelineState), &pipeline62);
    check(hr62 == S_OK && pipeline62, 16, "slot 62 hands out a pipeline");

    // Slot 86: a stream with a mesh (console type 24) and a pixel stage (type 2), 24-byte subobjects.
    const auto ms = container(2, 300), ps = container(3, 100);
    unsigned char stream[48]{};
    const auto put = [&](std::size_t at, std::uint32_t type, const std::vector<unsigned char>& code) {
        std::memcpy(stream + at, &type, 4);
        const D3D12_SHADER_BYTECODE bytecode{code.data(), code.size()};
        std::memcpy(stream + at + 8, &bytecode, sizeof(bytecode));
    };
    put(0, 24, ms);
    put(24, 2, ps);
    const D3D12_PIPELINE_STATE_STREAM_DESC stream_desc{sizeof(stream), stream};
    using Create86 = HRESULT(WINAPI*)(void*, const void*, const GUID*, void**);
    void* pipeline86 = nullptr;
    const HRESULT hr86 = reinterpret_cast<Create86>(slot(device, 86))(device, &stream_desc, &__uuidof(ID3D12PipelineState), &pipeline86);
    check(hr86 == S_OK && pipeline86, 16, "slot 86 hands out a pipeline");

    // The hook counters.
    long long total = 0;
    for (int i = 0; i < x2w_gpu::vc_count; ++i) total += x2w_gpu_counter(i);
    if (on) {
        check(x2w_gpu_counter(x2w_gpu::vc_pipelines) == 2, 32, "2 pipelines reached the ABI path");
        check(x2w_gpu_counter(x2w_gpu::vc_keys) == 3, 32, "3 keys logged");
        check(x2w_gpu_counter(x2w_gpu::vc_refused_57) == 1, 32, "1 refusal at slot 57");
        check(x2w_gpu_counter(x2w_gpu::vc_heaps) == 3, 32, "3 heaps reached the ABI path");
        check(x2w_gpu_counter(x2w_gpu::vc_copies) == 1, 32, "1 copy reached the ABI path");
        check(x2w_gpu_counter(x2w_gpu::vc_views) == 1, 32, "1 view reached the ABI path");
    } else {
        char what[64];
        std::snprintf(what, sizeof(what), "no hook counter moved (sum %lld)", total);
        check(total == 0, 32, what);
    }

    // The key file.
    std::ifstream file(keylog);
    if (on) {
        std::stringstream text;
        text << file.rdbuf();
        const std::string keys = text.str();
        std::printf("key file:\n%s", keys.c_str());
        for (const auto& [code, stage, slot_name] : {std::make_tuple(&cs, "compute", "slot62"), std::make_tuple(&ms, "mesh", "slot86"),
                                                      std::make_tuple(&ps, "pixel", "slot86")}) {
            const std::string line = key_of(code->data(), code->size()) + " " + stage + " " + slot_name + " " + std::to_string(code->size());
            check(keys.find(line) != std::string::npos, 64, line.c_str());
        }
    } else {
        check(!file.is_open(), 64, "no key file written");
    }
    file.close();
    DeleteFileA(keylog);
    return failed ? 10 + failed : 0;
}
