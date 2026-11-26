#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "startup_cx_async_cs.h"
#include "startup_cx_heap_cs.h"
#include "startup_cx_heap_bench_cs.h"
#include "startup_cx_heap_raw_cs.h"
#include "startup_cx_heap_words_cs.h"  // LOCAL isa-gaps
#include "startup_cx_sampler_cs.h"

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
using Word = std::uint64_t;
template <class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }
Word invoke(void* object, unsigned index, Word a = 0, Word b = 0, Word c = 0, Word d = 0, Word e = 0, Word f = 0)
{
    using Slot = Word (WINAPI*)(void*, Word, Word, Word, Word, Word, Word);
    return reinterpret_cast<Slot>((*static_cast<void***>(object))[index])(object, a, b, c, d, e, f);
}
bool ok(Word hr) { return SUCCEEDED(static_cast<HRESULT>(hr)); }
std::uint64_t fnv(const void* data, std::size_t bytes, std::uint64_t hash = 1469598103934665603ull)
{
    for (std::size_t i = 0; i < bytes; ++i) hash = (hash ^ static_cast<const unsigned char*>(data)[i]) * 1099511628211ull;
    return hash;
}
bool write_file(const std::wstring& path, const void* data, std::size_t bytes)
{
    FILE* f = _wfopen(path.c_str(), L"wb");
    if (!f) return false;
    const bool written = std::fwrite(data, 1, bytes, f) == bytes;
    return std::fclose(f) == 0 && written;
}
std::string read_log(const std::wstring& path)
{
    std::ifstream file(path, std::ios::binary);
    return std::string{std::istreambuf_iterator<char>(file), {}};
}
HANDLE hold_event;
DWORD WINAPI release_later(void*)
{
    Sleep(300);
    SetEvent(hold_event);
    return 0;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 3) return 2;
    const std::string mode = argv[2];
    if (mode != "async" && mode != "sync" && mode != "known" && mode != "seen" && mode != "heap" && mode != "sampler" && mode != "rawheap" &&
        mode != "rawheap-off" && mode != "rawheap-reset" && mode != "rawheap-late" && mode != "rawheap-bench" && mode != "slow" && mode != "slow-off" &&
        mode != "heapwords" && mode != "heapwords-raw")
        return 2;
    const bool slow = mode == "slow" || mode == "slow-off";
    const bool heapwords = mode.rfind("heapwords", 0) == 0;
    const bool heapwords_raw = mode == "heapwords-raw";  // LOCAL isa-gaps: `heapwords 1` (the rawheap heap, d3d12_vdesc.inl words)
    const bool rawheap = mode.rfind("rawheap", 0) == 0 || heapwords;  // LOCAL: `heapraw 1` (d3d12_heap_raw.inl)
    if (rawheap) {
        SetEnvironmentVariableW(L"X2W_BUFFER_HEAP_INDEX", L"1");
        SetEnvironmentVariableW(L"X2W_HEAP_RAW", mode == "rawheap-off" ? L"0" : nullptr);
        const bool reset_like = mode == "rawheap-reset" || mode == "rawheap-late";
        SetEnvironmentVariableW(L"X2W_HEAP_RAW_TEST_ENTRIES", reset_like ? L"16" : nullptr);
        SetEnvironmentVariableW(L"X2W_HEAP_RAW_TIMES", reset_like ? L"1" : nullptr);
        // The sweep thread's part waited for (results in the same frame, as an inline sweep): the late staging copy is
        // resolved by the drain of the frame that copied it. rawheap-bench runs the thread as a title run does.
        SetEnvironmentVariableW(L"X2W_HEAP_RAW_SWEEP_SYNC", mode == "rawheap-reset" ? L"1" : nullptr);
    }
    const bool bench = mode == "rawheap-bench";  // LOCAL: drain / eviction timings (X2W_BENCH_*: see the bench block)
    if (bench) {
        wchar_t entries[16] = L"16384";
        GetEnvironmentVariableW(L"X2W_BENCH_ENTRIES", entries, 16);  // 0: the runtime's own table size
        SetEnvironmentVariableW(L"X2W_HEAP_RAW_TEST_ENTRIES", std::wcscmp(entries, L"0") ? entries : nullptr);
        SetEnvironmentVariableW(L"X2W_HEAP_RAW_TIMES", L"1");
    }
    const bool sampler = mode == "sampler";
    if (sampler) SetEnvironmentVariableW(L"X2W_SAMPLER_LOD", L"1");
    const bool async = mode == "async";
    const bool heap = mode == "heap" || rawheap;  // LOCAL meshrt: `heap 1`, HeapMain, a 4-slot title view heap (no hold)
    const bool prewarm = mode == "known" || mode == "seen";  // X2W_CX_PREWARM_ALL (no hold)
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    ComPtr<IDXGIAdapter> adapter;
    if (std::strcmp(argv[1], "hardware") == 0) {
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
    } else if (std::strcmp(argv[1], "warp") != 0 || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) {
        return 3;
    }
    DXGI_ADAPTER_DESC adapter_desc{};
    LARGE_INTEGER driver{};
    if (FAILED(adapter->GetDesc(&adapter_desc))) return 3;
    adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driver);
    std::printf("adapter=%ls driver=%u.%u.%u.%u mode=%s\n", adapter_desc.Description, HIWORD(driver.HighPart), LOWORD(driver.HighPart),
                HIWORD(driver.LowPart), LOWORD(driver.LowPart), mode.c_str());
    {  // existing heaps over process memory (the windows); a probe device only
        ComPtr<ID3D12Device3> desktop;
        if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&desktop)))) return 4;
        void* block = VirtualAlloc(nullptr, 65536, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        ComPtr<ID3D12Heap> heap;
        const HRESULT hr = block ? desktop->OpenExistingHeapFromAddress(block, IID_PPV_ARGS(&heap)) : E_OUTOFMEMORY;
        heap.Reset();
        if (block) VirtualFree(block, 0, MEM_RELEASE);
        if (FAILED(hr)) {
            std::printf("OpenExistingHeapFromAddress -> 0x%08lx: capability skip\n", static_cast<unsigned long>(hr));
            return 77;
        }
    }

    // Shader cache, log and hold event of this process (process id + tick count: never an earlier run's files).
    wchar_t temp[MAX_PATH];
    if (!GetTempPathW(MAX_PATH, temp)) return 5;
    const std::wstring tag = std::to_wstring(GetCurrentProcessId()) + L"_" + std::to_wstring(GetTickCount64());
    const std::wstring cache = std::wstring(temp) + L"x2w_cx_async_" + tag;
    CreateDirectoryW(cache.c_str(), nullptr);
    SetEnvironmentVariableW(L"X2W_SHADER_CACHE", cache.c_str());
    const std::wstring log_path = cache + L"\\trap.log";
    SetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path.c_str());
    const std::wstring event_name = L"Local\\x2w_cx_hold_" + tag;
    hold_event = CreateEventW(nullptr, TRUE, FALSE, event_name.c_str());
    if (!hold_event) return 5;
    if (!prewarm && !heap && !sampler && !slow) SetEnvironmentVariableW(L"X2W_CX_TEST_HOLD_EVENT", event_name.c_str());

    // The title's compute root signature (its bytes go into the key; the translated one comes from the manifest).
    D3D12_ROOT_PARAMETER title_params[2]{};
    D3D12_DESCRIPTOR_RANGE title_range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
    title_params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    title_params[1].DescriptorTable = {1, &title_range};
    title_params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    const D3D12_ROOT_SIGNATURE_DESC title_root_desc{sampler ? 2u : 1u, title_params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> title_blob;
    if (FAILED(D3D12SerializeRootSignature(&title_root_desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &title_blob, nullptr))) return 5;
    static unsigned char code[192];  // fake console code (the stand-in only hashes it)
    for (int i = 0; i < 192; ++i) code[i] = static_cast<unsigned char>(0x3B ^ (i * 5));
    const std::uint64_t key = (fnv(title_blob->GetBufferPointer(), title_blob->GetBufferSize(), fnv(code, sizeof(code))) & 0x00ffffffffffffffull) |
                              (0x67ull << 56);
    char key_text[24];
    std::snprintf(key_text, sizeof(key_text), "%016llx", static_cast<unsigned long long>(key));
    std::printf("compute key %s\n", key_text);
    // known / seen: a second cached translation (key B, never created or dispatched); seen: a keys-seen file of
    // B and C (C has no translation), so only B is queued at start and the file afterwards holds A, B and C.
    const std::uint64_t key_b = key ^ 0x10, key_c = 0x67000000000000c0ull;
    // slow / slow-off: S (slow, 6 s more) and T (normal) sort before key A (same size, smaller keys), so the one
    // background worker takes S first, T and A wait behind it.
    const std::uint64_t key_s = 0x6700000000000001ull, key_t = 0x6700000000000002ull;
    char s_text[24], t_text[24];
    std::snprintf(s_text, sizeof(s_text), "%016llx", static_cast<unsigned long long>(key_s));
    std::snprintf(t_text, sizeof(t_text), "%016llx", static_cast<unsigned long long>(key_t));
    if (slow) {
        const std::string slow_spec = std::string(s_text) + ":6000";
        SetEnvironmentVariableA("X2W_CX_TEST_SLOW_KEY", slow_spec.c_str());
    }
    for (const std::uint64_t k : {key, key_b, key_s, key_t}) {
        if (k == key_b && !prewarm) continue;
        if ((k == key_s || k == key_t) && !slow) continue;
        wchar_t name[24];
        swprintf(name, 24, L"%016llx", static_cast<unsigned long long>(k));
        const std::wstring base = cache + L"\\" + name;
        const std::string manifest = sampler ? "cs 1 1 1\nuav 0 buf root 0 0\nsrv 1 tex2d table 1 0\ncssampler 1\n"
                                     : heapwords_raw ? "cs 1 1 1\nuav 0 buf root 0 0\nheap 1\nheapraw 1\nheapwords 1\n"
                                     : heapwords ? "cs 1 1 1\nuav 0 buf root 0 0\nheap 1\nheapwords 1\n"
                                     : rawheap ? "cs 1 1 1\nuav 0 buf root 0 0\nheap 1\nheapraw 1\n"
                                     : heap ? "cs 1 1 1\nuav 0 buf root 0 0\nheap 1\n" : "cs 1 1 1\nuav 0 buf root 0 0\n";
        if (!(sampler ? write_file(base + L".cs.cso", startup_cx_sampler_cs, sizeof(startup_cx_sampler_cs))
              : bench ? write_file(base + L".cs.cso", startup_cx_heap_bench_cs, sizeof(startup_cx_heap_bench_cs))
              : heapwords ? write_file(base + L".cs.cso", startup_cx_heap_words_cs, sizeof(startup_cx_heap_words_cs))
              : rawheap ? write_file(base + L".cs.cso", startup_cx_heap_raw_cs, sizeof(startup_cx_heap_raw_cs))
              : heap ? write_file(base + L".cs.cso", startup_cx_heap_cs, sizeof(startup_cx_heap_cs))
                   : write_file(base + L".cs.cso", startup_cx_async_cs, sizeof(startup_cx_async_cs))) ||
            !write_file(base + L".cs.txt", manifest.data(), manifest.size()))
            return 5;
    }
    const auto key_line = [](std::uint64_t k) {
        char text[24];
        std::snprintf(text, sizeof(text), "%016llx\n", static_cast<unsigned long long>(k));
        return std::string(text);
    };
    const std::wstring seen_path = cache + L"\\cx_keys_seen.lst";
    if (mode == "seen") {
        const std::string seen = key_line(key_b) + key_line(key_c);
        if (!write_file(seen_path, seen.data(), seen.size())) return 5;
    }

    std::array<unsigned char, 64> device_parameters{};
    void* device = nullptr;
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), device_parameters.data(), __uuidof(ID3D12Device), &device))) return 6;
    ComPtr<ID3D12Device> desktop;  // the same device object as the stand-in's (one device per adapter)
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&desktop)))) return 6;
    // known / seen: the start prewarm loads the known keys before the title creates anything (at most 15 s).
    const std::string all_loaded = std::string("x2w-cx prewarm: all ") + (mode == "known" ? "2" : "1") + " known pipelines loaded";
    if (prewarm) {
        bool loaded = false;
        for (int i = 0; i < 300 && !loaded; ++i) {
            loaded = read_log(log_path).find(all_loaded) != std::string::npos;
            if (!loaded) Sleep(50);
        }
        std::printf("start prewarm: \"%s\" %s\n", all_loaded.c_str(), loaded ? "seen" : "NOT seen within 15 s");
        if (!loaded) return 9;
    }
    void *title_root = nullptr, *pipeline = nullptr;
    if (!ok(invoke(device, 16, 0, arg(title_blob->GetBufferPointer()), title_blob->GetBufferSize(), arg(&__uuidof(ID3D12RootSignature)), arg(&title_root))))
        return 7;
    // The slot-67 record: [0] root signature, [5]/[6] code. Creation queues the key for the prewarm workers, whose
    // load now waits for the event.
    alignas(8) static unsigned char record[0x300];
    Word record_words[9] = {};
    record_words[0] = arg(title_root);
    record_words[5] = arg(code), record_words[6] = sizeof(code);
    std::memcpy(record, record_words, sizeof(record_words));
    if (!ok(invoke(device, 67, arg(record), arg(&__uuidof(ID3D12PipelineState)), arg(&pipeline))) || !pipeline) return 8;

    // Title memory: 2 MiB, word 0 the counter; a placed buffer over all of it (the window the UAV is made on; a root
    // UAV is bound over the readable bytes from its address, at most 1 MiB, which must lie inside one window).
    auto* memory = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, 2u << 20, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (!memory) return 10;
    D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, 2u << 20, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    void* placed = nullptr;
    if (!ok(invoke(device, 133, arg(memory), arg(&buffer), D3D12_RESOURCE_STATE_COMMON, 0, arg(&__uuidof(ID3D12Resource)), arg(&placed)))) return 11;

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    const unsigned list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    void *queue = nullptr, *fence = nullptr, *allocator = nullptr, *list = nullptr;
    if (!ok(invoke(device, 8, arg(&queue_desc), arg(&__uuidof(ID3D12CommandQueue)), arg(&queue))) ||
        !ok(invoke(device, 36, 0, 0, arg(&__uuidof(ID3D12Fence)), arg(&fence))) ||
        !ok(invoke(device, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, arg(&__uuidof(ID3D12CommandAllocator)), arg(&allocator))) ||
        !ok(invoke(device, 73, arg(list_desc), arg(allocator), 0, arg(&__uuidof(ID3D12GraphicsCommandList)), arg(&list))))
        return 13;
    void *sampler_heap = nullptr, *image_heap = nullptr;
    Word fence_value = 0;
    // One list of `dispatches` Dispatch(1, 1, 1), executed and waited for.
    const auto run_list = [&](int dispatches) {
        invoke(list, 139, arg(allocator));            // Reset
        if (image_heap) {
            void* heaps[]{image_heap, sampler_heap};
            invoke(list, 32, sampler_heap ? 2 : 1, arg(heaps));
        }
        invoke(list, 33, arg(title_root));            // SetComputeRootSignature
        if (image_heap) invoke(list, 35, 1, invoke(image_heap, 10));
        invoke(list, 25, arg(pipeline));              // SetPipelineState (the slot-67 pipeline)
        invoke(list, 45, 0, arg(memory));             // SetComputeRootUnorderedAccessView(0, title address)
        for (int i = 0; i < dispatches; ++i) invoke(list, 14, 1, 1, 1);
        invoke(list, 9);                              // Close
        void* lists[] = {list};
        invoke(queue, 10, 1, arg(lists));
        if (!ok(invoke(queue, 14, arg(fence), ++fence_value))) return false;
        const auto deadline = GetTickCount64() + 30000;
        while (invoke(fence, 8) < fence_value) {
            if (GetTickCount64() >= deadline) return false;
            Sleep(1);
        }
        return true;
    };

    int failures = 0;
    if (sampler) {
        const struct { D3D12_DESCRIPTOR_HEAP_DESC desc; Word backing, bytes; } views{
            {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, arg(memory + 0x50000), 32};
        if (!ok(invoke(device, 14, arg(&views), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&image_heap)))) return 12;
        const D3D12_RESOURCE_DESC image{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 2, 2, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                                        {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN};
        void* texture = nullptr;
        auto* pixels = VirtualAlloc(nullptr, 65536, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!pixels || !ok(invoke(device, 133, arg(pixels), arg(&image), D3D12_RESOURCE_STATE_COMMON, 0,
                                  arg(&__uuidof(ID3D12Resource)), arg(&texture)))) return 12;
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{image.Format, D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
        srv.Texture2D.MipLevels = 1;
        invoke(device, 18, arg(texture), arg(&srv), invoke(image_heap, 9));
        if (!run_list(1) || memory[0] != 0) return 16;
        auto* words = memory + 0x40000;
        const struct {
            D3D12_DESCRIPTOR_HEAP_DESC desc;
            Word backing, bytes;
        } hd{{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, arg(words), 32};
        if (!ok(invoke(device, 14, arg(&hd), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&sampler_heap))) || !sampler_heap) return 12;
        words[4] = 0x1ff;  // BORDER on U, V and W
        words[7] = 2u << 30;
        if (!run_list(1) || memory[0] != 255) return 16;
        words[4] = 0x56573258; words[5] = UINT_MAX;
        if (!run_list(1) || memory[0] != 255) return 16;
        std::puts("PASS: compute sampled the white border; unbound and invalid-token dispatches refused");
        return 0;
    }
    if (bench) {
        using u64 = std::uint64_t;
        const auto env = [](const wchar_t* name, unsigned long fallback) {
            wchar_t v[32] = L"";
            const DWORD n = GetEnvironmentVariableW(name, v, 32);
            return n && n < 32 ? std::wcstoul(v, nullptr, 10) : fallback;
        };
        const unsigned long frames = env(L"X2W_BENCH_FRAMES", 40), ranges = env(L"X2W_BENCH_RANGES", 300),
                            vslots = env(L"X2W_BENCH_VSLOTS", 2000), slots = env(L"X2W_BENCH_SLOTS", 65536),
                            period = env(L"X2W_BENCH_PERIOD", 0), blocks = env(L"X2W_BENCH_BLOCKS", 4);
        if (vslots + 8 > slots || !ranges || !blocks) return 2;
        const u64 heap_bytes = u64{slots} * 32;
        void* heap_memory = VirtualAlloc(nullptr, static_cast<SIZE_T>(heap_bytes), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        const struct {
            D3D12_DESCRIPTOR_HEAP_DESC desc;
            Word backing, bytes;
        } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, static_cast<UINT>(slots), D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE},
                    arg(heap_memory), heap_bytes};
        void* view_heap = nullptr;
        if (!heap_memory || !ok(invoke(device, 14, arg(&heap_desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&view_heap))) || !view_heap)
            return 12;
        constexpr u64 block_bytes = 32ull << 20;
        std::vector<unsigned char*> data;
        for (unsigned long b = 0; b < blocks; ++b) {
            data.push_back(static_cast<unsigned char*>(VirtualAlloc(nullptr, block_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)));
            if (!data.back()) return 12;
            auto* d = reinterpret_cast<std::uint32_t*>(data.back());  // dword i: its own address >> 2 (the program checks it)
            for (u64 i = 0; i < block_bytes / 4; ++i) d[i] = static_cast<std::uint32_t>((reinterpret_cast<u64>(d + i)) >> 2);
        }
        const Word cpu = invoke(view_heap, 9);
        D3D12_RESOURCE_DESC rdesc{};
        rdesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rdesc.Width = block_bytes;
        rdesc.Height = rdesc.DepthOrArraySize = rdesc.MipLevels = rdesc.SampleDesc.Count = 1;
        rdesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        struct ViewAt {
            D3D12_SHADER_RESOURCE_VIEW_DESC view;
            Word address;
            Word tail[2];
        } v{};
        v.view = {DXGI_FORMAT_R32_TYPELESS, D3D12_SRV_DIMENSION_BUFFER, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
        v.view.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        const auto mix = [](std::uint64_t x) {
            x ^= x >> 33;
            x *= 0xff51afd7ed558ccdull;
            x ^= x >> 33;
            return x;
        };
        u64 cursor = 0, slot = 0;  // byte cursor over the blocks, V# slot ring
        const u64 space = block_bytes * blocks;
        unsigned char planes[0x50] = {}, present[0x40] = {};
        LARGE_INTEGER f, t0, t1;
        QueryPerformanceFrequency(&f);
        double write_ms = 0;
        const ULONGLONG started = GetTickCount64();
        unsigned long ring_checks = 0, ring_bad = 0;
        for (unsigned long frame = 0; frame < frames; ++frame) {
            if (period && frame % period == 0) cursor = 0;  // ring: the same ranges again
            const u64 salt = period ? frame % period : frame;
            QueryPerformanceCounter(&t0);
            // this frame's V#s; every 10th frame and the last check the whole ring (V#s of earlier frames, kept by the sweep)
            const bool whole = frame + 1 == frames || frame % 10 == 9;
            memory[1] = static_cast<std::uint32_t>(whole ? 0 : slot);
            memory[2] = static_cast<std::uint32_t>(whole ? vslots : (std::min<u64>)(ranges, vslots));
            const std::uint32_t bad_before = memory[5];
            memory[3] = static_cast<std::uint32_t>(vslots);
            for (unsigned long i = 0; i < ranges; ++i) {
                const u64 bytes = 16 * (1 + mix(salt * 1000003ull + i) % 256);
                if (cursor % block_bytes + bytes > block_bytes) cursor = (cursor / block_bytes + 1) * block_bytes;
                if (cursor + bytes > space) cursor = 0;
                v.view.Buffer.NumElements = static_cast<UINT>(bytes / 4);
                v.address = arg(data[cursor / block_bytes] + cursor % block_bytes);
                invoke(device, 70, arg(&rdesc), arg(&v), cpu + 32 * (8 + slot));
                cursor = (cursor + bytes + 255) & ~255ull;
                slot = (slot + 1) % vslots;
            }
            QueryPerformanceCounter(&t1);
            write_ms += static_cast<double>(t1.QuadPart - t0.QuadPart) * 1000.0 / static_cast<double>(f.QuadPart);
            if (!run_list(1)) return 14;
            if (whole) {
                ++ring_checks;
                ring_bad += memory[5] - bad_before;
            }
            invoke(queue, 32, 0, arg(planes), arg(present));
        }
        const double wall = (GetTickCount64() - started) / 1000.0;
        const std::string log = read_log(log_path);
        if (log.find(std::string("x2w-cx ") + key_text + ": heap dispatch skipped: increment not 32") != std::string::npos) {
            std::printf("heap dispatches need a 32-byte view increment: capability skip\n");
            return 77;
        }
        // The drain and eviction times the runtime logged (X2W_HEAP_RAW_TIMES=1).
        const auto times = [&](const char* prefix, const char* number_after) {
            std::vector<double> ms;
            for (std::size_t p = log.find(prefix); p != std::string::npos; p = log.find(prefix, p + 1)) {
                const std::size_t end = log.find('\n', p);
                const std::string line = log.substr(p, end - p);
                const std::size_t at = line.find(number_after);
                if (at != std::string::npos) ms.push_back(std::strtod(line.c_str() + at + std::strlen(number_after), nullptr));
            }
            return ms;
        };
        const auto stats = [](const char* what, std::vector<double> ms, std::size_t skip) {
            if (ms.size() <= skip) {
                std::printf("%s: %zu samples\n", what, ms.size());
                return 0.0;
            }
            ms.erase(ms.begin(), ms.begin() + skip);
            std::vector<double> sorted = ms;
            std::sort(sorted.begin(), sorted.end());
            double sum = 0;
            for (double x : ms) sum += x;
            std::printf("%s: %zu samples (first %zu skipped), mean %.3f ms, p50 %.3f, p95 %.3f, max %.3f\n", what, ms.size(), skip,
                        sum / ms.size(), sorted[sorted.size() / 2], sorted[sorted.size() * 95 / 100], sorted.back());
            return sorted.back();
        };
        const std::vector<double> drains = times("x2w-heapraw drain time: ", "drain time: ");
        const std::vector<double> work = times("x2w-heapraw frame work: ", "frame work: ");
        std::printf("bench: %lu frames x %lu ranges (%s), %lu V# slots in a %lu-slot heap, %lu x 32 MiB data blocks; %.1f s wall, "
                    "slot-70 writes %.3f ms per frame\n",
                    frames, ranges, period ? "ring" : "fresh", vslots, slots, blocks, wall, write_ms / frames);
        stats("drains (eviction included)", drains, 1);
        stats("frame work (sweep and eviction, inside the drain)", work, 1);
        for (const char* line : {"x2w-heapraw ranges:", "x2w-heapraw drain time", "x2w-heapraw frame work", "x2w-heapraw drains with work", "x2w-d3d12 time inside"}) {
            const std::size_t p = log.rfind(line);
            if (p != std::string::npos) std::printf("last: %s\n", log.substr(p, log.find('\n', p) - p).c_str());
        }
        const bool counted = drains.size() == frames && (!frames || log.find("x2w-heapraw ranges:") != std::string::npos);
        std::printf("drain lines %zu of %lu frames\n", drains.size(), frames);
        // Correctness: every checked V# read its data (none missed or wrong), and the page cache agreed with
        // desktop_range for every range (X2W_HEAP_RAW_CHECK=1).
        const std::size_t c = log.rfind("check mismatches ");
        const long mismatches = c == std::string::npos ? -1 : std::strtol(log.c_str() + c + 17, nullptr, 10);
        std::printf("reads through V# slots: %u checked, %u wrong or missed; whole-ring reads: %lu of %lu slots each, %lu wrong or missed "
                    "in all; page cache check mismatches %ld\n",
                    memory[6], memory[5], ring_checks, vslots, ring_bad, mismatches);
        // A miss is allowed only as a counted displacement of a recently seen entry (all probes occupied; the sweep
        // brings it back within a pass), never from eviction.
        const std::size_t d = log.rfind("(recent ");
        const long recent = d == std::string::npos ? -1 : std::strtol(log.c_str() + d + 8, nullptr, 10);
        std::printf("displaced recent entries %ld\n", recent);
        if (static_cast<long>(memory[5]) > recent || memory[6] == 0 || mismatches != 0) ++failures;
        if (desktop->GetDeviceRemovedReason() != S_OK || !counted || failures) return 16;
        std::printf("PASS (timings are reported, not checked)\n");
        return 0;
    }
    if (rawheap) {
        // A shader-visible 8-slot title view heap at title + 1 MiB (256 bytes); two private 64 KiB blocks no resource
        // covers (two desktop windows): data (dwords 0xc0de0000 + i) and data2 (0xd2000000 + i).
        const struct {
            D3D12_DESCRIPTOR_HEAP_DESC desc;
            Word backing, bytes;
        } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 10, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, arg(memory + 0x40000), 10 * 32};
        void* view_heap = nullptr;
        if (!ok(invoke(device, 14, arg(&heap_desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&view_heap))) || !view_heap) return 12;
        auto* data = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, 65536, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
        auto* data2 = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, 65536, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
        if (!data || !data2) return 12;
        for (std::uint32_t i = 0; i < 64; ++i) data[i] = 0xc0de0000u + i, data2[i] = 0xd2000000u + i;
        // Two adjacent private 64 KiB blocks (two allocations, so two desktop windows): pa dwords 0xa0000000 + i, pb
        // 0xb000b000 + i. Slot 7's V# spans both (a paged entry).
        auto* pa = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, 2 * 65536, MEM_RESERVE, PAGE_NOACCESS));
        if (!pa || !VirtualFree(pa, 0, MEM_RELEASE)) return 12;
        pa = static_cast<std::uint32_t*>(VirtualAlloc(pa, 65536, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
        auto* pb = pa ? static_cast<std::uint32_t*>(VirtualAlloc(pa + 16384, 65536, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)) : nullptr;
        if (!pa || !pb) return 12;
        for (std::uint32_t i = 0; i < 16384; ++i) pa[i] = 0xa0000000u + i, pb[i] = 0xb000b000u + i;
        const bool late = mode == "rawheap-late";  // the sweep thread as in a title run (no SYNC)
        const bool reset = mode == "rawheap-reset" || late;
        const bool on = mode != "rawheap-off";
        const Word cpu = invoke(view_heap, 9);  // GetCPUDescriptorHandleForHeapStart: the title's slot 0
        D3D12_RESOURCE_DESC rdesc{};
        rdesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rdesc.Width = 65536;
        rdesc.Height = rdesc.DepthOrArraySize = rdesc.MipLevels = rdesc.SampleDesc.Count = 1;
        rdesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        struct ViewAt {
            D3D12_SHADER_RESOURCE_VIEW_DESC view;
            Word address;
            Word tail[2];
        } v{};
        static_assert(offsetof(ViewAt, address) == 0x28 && sizeof(ViewAt) == 0x40);
        const D3D12_SHADER_RESOURCE_VIEW_DESC raw16 = [] {
            D3D12_SHADER_RESOURCE_VIEW_DESC d{DXGI_FORMAT_R32_TYPELESS, D3D12_SRV_DIMENSION_BUFFER, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
            d.Buffer.NumElements = 4;
            d.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            return d;
        }();
        const auto memory_srv = [&](const D3D12_SHADER_RESOURCE_VIEW_DESC& view, Word address, int slot) {
            v.view = view;
            v.address = address;
            invoke(device, 70, arg(&rdesc), arg(&v), cpu + 32 * slot);
        };
        memory_srv(raw16, arg(data), 1);       // slot 1: the runtime writes the V#
        memory_srv(raw16, arg(data) + 2, 3);   // slot 3: base not 4-aligned
        memory_srv(raw16, arg(pa), 8);         // slots 8 and 9: one window each
        memory_srv(raw16, arg(pb), 9);
        D3D12_SHADER_RESOURCE_VIEW_DESC raw32 = raw16;
        raw32.Buffer.NumElements = 8;
        memory_srv(raw32, arg(pa) + 0xfff0, 7);  // slot 7: 32 bytes over both blocks
        D3D12_SHADER_RESOURCE_VIEW_DESC structured{DXGI_FORMAT_UNKNOWN, D3D12_SRV_DIMENSION_BUFFER, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
        structured.Buffer.NumElements = 3;
        structured.Buffer.StructureByteStride = 12;
        memory_srv(structured, arg(data2), 4);  // slot 4: structured, 3 x 12 bytes, the second window
        D3D12_SHADER_RESOURCE_VIEW_DESC typed{DXGI_FORMAT_R32_UINT, D3D12_SRV_DIMENSION_BUFFER, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
        typed.Buffer.NumElements = 4;
        memory_srv(typed, arg(data) + 128, 5);  // slot 5: typed R32_UINT, 4 elements (data dwords 32..35)
        // Slot 6: a view token, a raw SRV (slot 18) of the title buffer at byte 0xc0000, dword 0x70000001.
        memory[0x30000] = 0x70000001u;
        D3D12_SHADER_RESOURCE_VIEW_DESC token = raw16;
        token.Buffer.FirstElement = 0x30000;
        invoke(device, 18, arg(placed), arg(&token), cpu + 32 * 6);
        auto* slots = reinterpret_cast<unsigned char*>(cpu);
        const auto* w1 = reinterpret_cast<const std::uint32_t*>(slots + 32);
        const auto* w4 = reinterpret_cast<const std::uint32_t*>(slots + 128);
        const auto* w5 = reinterpret_cast<const std::uint32_t*>(slots + 160);
        const auto* w6 = reinterpret_cast<const std::uint32_t*>(slots + 192);
        std::printf("slot 1 words %08x %08x %08x %08x; slot 4 %08x %08x %08x %08x; slot 5 %08x %08x %08x %08x; slot 6 %08x %08x\n", w1[0],
                    w1[1], w1[2], w1[3], w4[0], w4[1], w4[2], w4[3], w5[0], w5[1], w5[2], w5[3], w6[0], w6[1]);
        for (int i = 4; i < 32; ++i) memory[i] = 0x11111111u;
        if (!reset) {
            std::memcpy(slots + 64, slots + 32, 32);  // slot 2: the title's own copy
            if (heapwords) {  // slot 3: a view token whose id has no view (the sentinel), then words never to be returned
                const std::uint32_t bad[8] = {0x56573258u, 0xffffffffu, 0xbad00002u, 0xbad00003u,
                                              0xbad00004u, 0xbad00005u, 0xbad00006u, 0xbad00007u};
                std::memcpy(slots + 96, bad, 32);
            }
            if (!run_list(1)) return 14;
        }
        unsigned char first_v[32] = {};
        for (Word k = 1; reset && k <= 4; ++k) {  // slot 2: a new V# before each dispatch, one frame each
            unsigned char planes[0x50] = {}, present[0x40] = {};
            if (k > 1) invoke(queue, 32, 0, arg(planes), arg(present));
            memory_srv(raw16, arg(data) + 16 * k, 2);
            if (k == (late ? 3 : 1)) std::memcpy(first_v, slots + 64, 32);
            if (!run_list(1)) return 14;
        }
        if (reset) {  // a late staging copy: an earlier slot-2 V# copied back with no note
            unsigned char planes[0x50] = {}, present[0x40] = {};
            invoke(queue, 32, 0, arg(planes), arg(present));
            std::memcpy(slots + 64, first_v, 32);
            if (!run_list(1)) return 14;
            if (late) {  // the thread's part of the copy frame is taken by the next frame's drain
                invoke(queue, 32, 0, arg(planes), arg(present));
                if (!run_list(1)) return 14;
            }
        }
        if (!reset) {  // two empty console presents (queue slot 32) 5 s apart reach the heartbeat (report_heap_raw)
            unsigned char planes[0x50] = {}, present[0x40] = {};
            invoke(queue, 32, 0, arg(planes), arg(present));
            Sleep(5200);
            invoke(queue, 32, 0, arg(planes), arg(present));
        }
        const std::string log = read_log(log_path);
        const auto has = [&](const std::string& text) { return log.find(text) != std::string::npos; };
        if (has(std::string("x2w-cx ") + key_text + ": heap dispatch skipped: increment not 32")) {
            std::printf("heap dispatches need a 32-byte view increment: capability skip\n");
            return memory[4] == 0x11111111u ? 77 : 16;
        }
        if (heapwords) {
            const std::uint32_t want[12] = {w1[0], 16u, 0x30016facu, 16u, 0x30016facu, 0u, 0u, 0u, 17u, 0u, 16u, w1[3]};
            const char* what[12] = {"slot 1 word 0", "slot 1 word 2", "slot 1 word 3", "slot 2 word 2", "slot 2 word 3",
                                    "slot 3 (token id 0xffffffff) word 0", "slot 3 word 2", "slot 0 word 0", "width idiom on slot 1",
                                    "slot 6 (token) word 1 (printed only)", "slot 6 word 2 (vdesc)", "slot 6 word 3 (vdesc)"};
            for (int i = 0; i < 12; ++i) {
                std::printf("%s: %08x (want %08x)\n", what[i], memory[4 + i], want[i]);
                if (i != 9 && memory[4 + i] != want[i]) ++failures;
            }
            const bool vdesc_line = has("x2w-vdesc buffer ") && has(std::string("x2w-cx ") + key_text + ": heapwords 1: view descriptor words");
            std::printf("vdesc lines %d\n", vdesc_line);
            if (!vdesc_line) ++failures;
            const bool raw_line = has(std::string("x2w-cx ") + key_text + ": heapraw 1: raw V# table at t2 space1");
            std::printf("heapraw line %d (want %d)\n", raw_line, heapwords_raw);
            if (raw_line != heapwords_raw) ++failures;
            for (std::size_t p = log.find("x2w-vdesc"); p != std::string::npos; p = log.find("x2w-vdesc", p + 1)) {
                const std::size_t end = log.find('\n', p);
                std::printf("%s\n", log.substr(p, end == std::string::npos ? std::string::npos : end - p).c_str());
            }
            for (std::size_t p = log.find("x2w-d3d12-debug"); p != std::string::npos; p = log.find("x2w-d3d12-debug", p + 1)) {
                const std::size_t end = log.find('\n', p);
                const std::string line = log.substr(p, end == std::string::npos ? std::string::npos : end - p);
                std::printf("%s\n", line.c_str());
                if (line.find("sev=0 ") != std::string::npos || line.find("sev=1 ") != std::string::npos) ++failures;
            }
            std::printf("device removed reason: 0x%08lx\n", static_cast<unsigned long>(desktop->GetDeviceRemovedReason()));
            if (desktop->GetDeviceRemovedReason() != S_OK) ++failures;
            std::printf("%s\n", failures ? "FAIL" : "PASS");
            return failures ? 16 : 0;
        }
        const std::uint32_t s2 = late ? 0xcu : reset ? 0x4u : 0u;  // slot 2's first dword in data (reset: the copied-back V#)
        const std::uint32_t typed_format = (w5[3] >> 12) & 127;
        const std::uint32_t want[28] = {
            on ? 0xc0de0000u : 0u, on ? 0xc0de0003u : 0u, 0u, 0u,                                                // slot 1
            on ? 0xc0de0000u + s2 : 0u, on ? 0xc0de0003u + s2 : 0u, on ? 0x0001c0deu + (s2 << 16) : 0u, 0u,    // slot 2
            on ? 0x0001c0deu : 0u, 0x11111111u, 0x11111111u, 0x11111111u,                                        // slot 3
            on ? 0xd2000008u : 0u, 0u, on ? 12u : 0u, 0u,                                                        // slot 4
            on ? 0xc0de0020u : 0u, on ? 4u : 0u, on && typed_format != 22 ? typed_format : 0u, 0u,               // slot 5
            0x70000001u, on ? 0xc0de0000u : 0u, 0u, 0u,                                                          // 6, 1, 3
            on ? 0xa0003ffcu : 0u, on ? 0xa0003fffu : 0u, on ? 0xb000b000u : 0u, on ? 0xb000a000u : 0u};         // slot 7
        const char* what[28] = {"slot 1 dword 0", "slot 1 dword 3", "slot 1 dword 4 (past 16 bytes)", "slot 1 stride",
                                "slot 2 dword 0", "slot 2 dword 3", "slot 2 byte 2", "(unused)",
                                "slot 3 dword 0 (base not 4-aligned: bytes 2..5)", "(unused)", "(unused)", "(unused)",
                                "slot 4 byte 32 (structured 3 x 12)", "slot 4 byte 36 (past 36 bytes)", "slot 4 stride", "slot 4 format",
                                "slot 5 dword 0 (typed)", "slot 5 stride", "slot 5 format", "slot 5 byte 16 (past 4 elements)",
                                "slot 6 dword 0 (token)", "slot 1 dword 0 (same program)", "slot 3 stride (raw: 0)", "slot 3 format (raw: 0)",
                                "slot 7 dword 0 (paged, first block)", "slot 7 dword 3", "slot 7 dword 4 (second block)",
                                "slot 7 byte 14 (across the page boundary)"};
        for (int i = 0; i < 28; ++i) {
            if (std::strcmp(what[i], "(unused)") == 0) continue;
            std::printf("%s: %08x (want %08x)\n", what[i], memory[4 + i], want[i]);
            if (memory[4 + i] != want[i]) ++failures;
        }
        const bool v_written = w1[0] == static_cast<std::uint32_t>(arg(data)) && w1[2] == 16 && w1[3] == 0x30016facu &&
                               w4[2] == 3 && ((w4[1] >> 16) & 0x3fff) == 12 && w5[2] == 4 && w6[0] == 0x56573258u;
        const bool table_line = has("x2w-cx " + std::string(key_text) + ": heapraw 1: raw V# table at t2 space1");
        const bool two_windows = on ? has("x2w-heapraw window 3: ") && !has("x2w-heapraw window 4: ") : true;  // data, data2, pa, pb
        const bool counted =
            late ? has("x2w-heapraw sweep thread started") && !has("sweep thread started (X2W_HEAP_RAW_SWEEP_SYNC=1)") && log.find("displaced 0 (recent 0)", log.rfind("x2w-heapraw drain time:")) != std::string::npos
            : reset ? log.rfind("x2w-heapraw drain time:") != std::string::npos &&
                        log.find("evicted 4, displaced 0 (recent 0)", log.rfind("x2w-heapraw drain time:")) != std::string::npos
            : on ? has("(base not 4-aligned 1, paged 1); not resolved: base not 4-aligned 0, not in windows 0, ") &&
                       has("x2w-heapraw drains with work: 1, ") &&
                       has("table half full: reset 0 (evicted 0, displaced 0); resolved 7 new ranges, 0 waiting, 0 notes dropped")
                 : !has("x2w-heapraw ranges:") && !has("x2w-heapraw drains with work");
        std::printf("V#s and token written %d, heapraw line %d, two windows %d, counters %d\n", v_written, table_line, two_windows, counted);
        if (!v_written || !table_line || !two_windows || !counted) ++failures;
        for (std::size_t p = log.find("x2w-heapraw"); p != std::string::npos; p = log.find("x2w-heapraw", p + 1)) {
            const std::size_t end = log.find('\n', p);
            std::printf("%s\n", log.substr(p, end == std::string::npos ? std::string::npos : end - p).c_str());
        }
        for (std::size_t p = log.find("x2w-d3d12-debug"); p != std::string::npos; p = log.find("x2w-d3d12-debug", p + 1)) {
            const std::size_t end = log.find('\n', p);
            const std::string line = log.substr(p, end == std::string::npos ? std::string::npos : end - p);
            std::printf("%s\n", line.c_str());
            if (line.find("sev=0 ") != std::string::npos || line.find("sev=1 ") != std::string::npos) ++failures;
        }
        std::printf("device removed reason: 0x%08lx\n", static_cast<unsigned long>(desktop->GetDeviceRemovedReason()));
        if (desktop->GetDeviceRemovedReason() != S_OK) ++failures;
        std::printf("%s\n", failures ? "FAIL" : "PASS");
        return failures ? 16 : 0;
    }
    if (heap) {
        // A shader-visible 4-slot title view heap at title + 1 MiB (128 bytes); word 0 of slot 1 = A, and word 0 of
        // "slot 4", the first word past the heap, = B. heapmem is a bounded view of the heap, so the shader must
        // read A and 0. (An unbounded root SRV, the earlier binding, reads B there.)
        const struct {
            D3D12_DESCRIPTOR_HEAP_DESC desc;
            Word backing, bytes;
        } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, arg(memory + 0x40000), 4 * 32};
        void* view_heap = nullptr;
        if (!ok(invoke(device, 14, arg(&heap_desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&view_heap))) || !view_heap) return 12;
        constexpr std::uint32_t A = 0xa1a1a1a1u, B = 0xb2b2b2b2u;
        memory[0x40000 + 8] = A;   // slot 1 (byte 32)
        memory[0x40000 + 32] = B;  // byte 128: past the heap
        memory[1] = memory[2] = 0x11111111u;
        if (!run_list(1)) return 14;
        const std::string log = read_log(log_path);
        const auto has = [&](const std::string& text) { return log.find(text) != std::string::npos; };
        const bool skipped32 = has(std::string("x2w-cx ") + key_text + ": heap dispatch skipped: increment not 32");
        std::printf("heap: in range %08x (want %08x), past the end %08x (want 00000000; unbounded would read %08x); increment skip %d\n",
                    memory[1], A, memory[2], B, skipped32);
        if (skipped32) {  // WARP: views are not 32 bytes apart, so `heap 1` dispatches are skipped by design
            std::printf("heap dispatches need a 32-byte view increment: capability skip\n");
            return memory[1] == 0x11111111u && memory[2] == 0x11111111u ? 77 : 16;
        }
        const bool bounded = has(std::string("x2w-cx ") + key_text + ": bounded heap views: heapmem 128 bytes (title heap 128), vinfo ");
        std::printf("log: bounded heap views line %d\n", bounded);
        if (memory[1] != A || memory[2] != 0 || !bounded) {
            std::printf("FAIL: heapmem is not a bounded view of the 128-byte heap\n");
            ++failures;
        }
        for (std::size_t p = log.find("x2w-d3d12-debug"); p != std::string::npos; p = log.find("x2w-d3d12-debug", p + 1)) {
            const std::size_t end = log.find('\n', p);
            const std::string line = log.substr(p, end == std::string::npos ? std::string::npos : end - p);
            std::printf("%s\n", line.c_str());
            if (line.find("sev=0 ") != std::string::npos || line.find("sev=1 ") != std::string::npos) ++failures;
        }
        std::printf("device removed reason: 0x%08lx\n", static_cast<unsigned long>(desktop->GetDeviceRemovedReason()));
        if (desktop->GetDeviceRemovedReason() != S_OK) ++failures;
        std::printf("%s\n", failures ? "FAIL" : "PASS");
        return failures ? 16 : 0;
    }
    if (slow) {
        const bool lane = mode == "slow";
        // The first dispatch waits until the worker is inside S's load (the watch's "load running" line, after
        // X2W_CX_SLOW_LOG_MS = 500): a dispatch issued before the worker took S would demand A while it is still
        // queued, and a demanded key is taken first with or without the lane, so the control would not starve.
        const std::string s_running = std::string("x2w-cx ") + s_text + ": load running for ";
        bool s_loading = false;
        for (int i = 0; i < 200 && !s_loading; ++i) {  // at most 10 s
            s_loading = read_log(log_path).find(s_running) != std::string::npos;
            if (!s_loading) Sleep(50);
        }
        if (!s_loading) {
            std::printf("S's load never logged \"load running\" before the dispatch\n");
            return 15;
        }
        const ULONGLONG start = GetTickCount64();
        int lists = 0;
        while (memory[0] == 0 && GetTickCount64() - start < 20000) {
            if (!run_list(1)) return 14;
            ++lists;
            if (memory[0] == 0) Sleep(50);
        }
        const ULONGLONG a_ms = GetTickCount64() - start;
        std::printf("A ran after %llu ms (%d one-dispatch lists), counter %u\n", static_cast<unsigned long long>(a_ms), lists, memory[0]);
        const std::string s_ended = std::string("x2w-cx ") + s_text + ": slow load ended after ";
        const std::string s_translated = std::string("x2w-cx ") + s_text + " translated: ";
        std::string log;
        for (int i = 0; i < 300; ++i) {  // wait for S's load to end (at most 15 s)
            log = read_log(log_path);
            if (log.find(s_translated) != std::string::npos && (!lane || log.find(s_ended) != std::string::npos)) break;
            Sleep(50);
        }
        const auto at = [&](const std::string& text) { return log.find(text); };
        const std::size_t s_at = at(s_translated), t_at = at(std::string("x2w-cx ") + t_text + " translated: "),
                          a_at = at(std::string("x2w-cx ") + key_text + " translated: ");
        const bool running = at(std::string("x2w-cx ") + s_text + ": load running for ") != std::string::npos;
        const bool marked = at(std::string("x2w-cx ") + s_text + ": marked slow after ") != std::string::npos &&
                            at("a replacement worker started") != std::string::npos;
        const bool ended = at(s_ended) != std::string::npos;
        const bool lane_line = at("plus 1 demand-lane thread") != std::string::npos;
        const bool all_three = s_at != std::string::npos && t_at != std::string::npos && a_at != std::string::npos;
        std::printf("log: lane line %d, running line %d, marked slow %d, slow ended %d; translated order S %zd T %zd A %zd\n", lane_line, running,
                    marked, ended, static_cast<std::ptrdiff_t>(s_at), static_cast<std::ptrdiff_t>(t_at), static_cast<std::ptrdiff_t>(a_at));
        if (memory[0] != 1 || !all_three) ++failures;
        if (lane) {
            // A within 3 s (S's load takes >= 6 s), T before S, every watch line.
            if (a_ms >= 3000 || !(a_at < s_at) || !(t_at < s_at) || !running || !marked || !ended || !lane_line) ++failures;
        } else {
            // the starvation the lane removes: A and T only after S (and no slow mark: X2W_CX_SLOW_MS above the test) -- A is measured from S's "load running" line (0.5-0.8 s into its 6 s), so >= 4 s
            if (a_ms < 4000 || !(s_at < a_at) || !(s_at < t_at) || marked || lane_line || !running) ++failures;
        }
        for (std::size_t p = log.find("x2w-cx"); p != std::string::npos; p = log.find("x2w-cx", p + 1)) {
            const std::size_t end = log.find('\n', p);
            const std::string line = log.substr(p, end == std::string::npos ? std::string::npos : end - p);
            if (line.find("prewarm") != std::string::npos || line.find("slow") != std::string::npos || line.find("running") != std::string::npos ||
                line.find("translated") != std::string::npos)
                std::printf("%s\n", line.c_str());
        }
        for (std::size_t p = log.find("x2w-d3d12-debug"); p != std::string::npos; p = log.find("x2w-d3d12-debug", p + 1)) {
            const std::size_t end = log.find('\n', p);
            const std::string line = log.substr(p, end == std::string::npos ? std::string::npos : end - p);
            std::printf("%s\n", line.c_str());
            if (line.find("sev=0 ") != std::string::npos || line.find("sev=1 ") != std::string::npos) ++failures;
        }
        if (desktop->GetDeviceRemovedReason() != S_OK) ++failures;
        std::printf("%s\n", failures ? "FAIL" : "PASS");
        return failures ? 16 : 0;
    }
    if (mode == "sync") {
        if (const HANDLE t = CreateThread(nullptr, 0, release_later, nullptr, 0, nullptr)) CloseHandle(t);
        else return 13;
    }
    const ULONGLONG list1_start = GetTickCount64();
    if (!run_list(3)) return 14;
    const std::uint32_t after_first = memory[0];
    std::printf(prewarm ? "list 1 (3 dispatches, pipeline prewarmed at start): counter %u, recorded and executed in %llu ms\n"
                        : "list 1 (3 dispatches, pipeline held back): counter %u, recorded and executed in %llu ms\n", after_first,
                static_cast<unsigned long long>(GetTickCount64() - list1_start));
    if (after_first != (async ? 0u : 3u)) {
        std::printf("FAIL: list 1 should leave the counter at %u\n", async ? 0u : 3u);
        ++failures;
    }
    int tries = 0;
    if (async) {
        SetEvent(hold_event);
        while (memory[0] == 0 && tries < 200) {
            Sleep(50);
            ++tries;
            if (!run_list(1)) return 14;
        }
        std::printf("after the release: counter %u after %d one-dispatch lists\n", memory[0], tries);
        if (memory[0] != 1) {
            std::printf("FAIL: once ready, exactly one dispatch should have run (counter 1)\n");
            ++failures;
        }
    }

    // The log (the worker writes the ready line just after the load completes: wait for it at most 5 s).
    const std::string skip_line = std::string("x2w-cx ") + key_text + ": pipeline not loaded yet, dispatch skipped";
    const std::string ready_line = std::string("x2w-cx ") + key_text + " ready (X2W_CX_ASYNC)";
    std::string log = read_log(log_path);
    for (int i = 0; async && i < 100 && log.find(ready_line) == std::string::npos; ++i) {
        Sleep(50);
        log = read_log(log_path);
    }
    const auto has = [&](const std::string& text) { return log.find(text) != std::string::npos; };
    long skipped = -1;
    if (const std::size_t p = log.find(ready_line); p != std::string::npos) {
        const std::string line = log.substr(p, log.find('\n', p) - p);
        std::printf("log: %s\n", line.c_str());
        if (const std::size_t q = line.find("; "); q != std::string::npos) skipped = std::strtol(line.c_str() + q + 2, nullptr, 10);
    }
    const bool translated = has(std::string("x2w-cx ") + key_text + " translated: ");
    const bool released = has(std::string("x2w-cx ") + key_text + ": load held by X2W_CX_TEST_HOLD_EVENT (released)");
    std::printf("log: translated %d, hold released %d, skip line %d, ready line %d (%ld dispatches skipped while compiling)\n", translated, released,
                has(skip_line), has(ready_line), skipped);
    if (!translated || released == prewarm) {
        std::printf("FAIL: the translation was not loaded %s\n", prewarm ? "(or the hold was set)" : "through the hold");
        ++failures;
    }
    if (prewarm) {
        // The queued line, and the keys-seen file the writer thread writes every 10 s (wait at most 25 s).
        const std::string queued = mode == "known"
                                       ? "x2w-cx prewarm: 2 known pipelines queued (X2W_CX_PREWARM_ALL; no keys seen file: every cached translation)"
                                       : "x2w-cx prewarm: 1 known pipelines queued (X2W_CX_PREWARM_ALL; keys seen file ";
        const std::string counts = ": 2 keys, 1 with a translation)";
        const bool queued_ok = has(queued) && (mode == "known" || has(counts));
        std::printf("log: queued line %d\n", queued_ok);
        if (!queued_ok) {
            std::printf("FAIL: the start prewarm did not queue what mode %s expects\n", mode.c_str());
            ++failures;
        }
        std::uint64_t sorted[3] = {key_c, key < key_b ? key : key_b, key < key_b ? key_b : key};  // key_c is the smallest
        const std::string want = mode == "known" ? key_line(key) : key_line(sorted[0]) + key_line(sorted[1]) + key_line(sorted[2]);
        std::string seen;
        for (int i = 0; i < 250; ++i) {
            seen = read_log(seen_path);
            if (seen == want) break;
            Sleep(100);
        }
        std::printf("keys seen file: \"%s\" (want \"%s\")\n", seen.c_str(), want.c_str());
        if (seen != want) {
            std::printf("FAIL: the keys seen file is not the union of the earlier file and the dispatched key\n");
            ++failures;
        }
    }
    if (async ? (!has(skip_line) || skipped < 3) : (has(skip_line) || has(ready_line))) {
        std::printf("FAIL: the X2W_CX_ASYNC lines do not match mode %s\n", mode.c_str());
        ++failures;
    }
    for (std::size_t p = log.find("x2w-d3d12-debug"); p != std::string::npos; p = log.find("x2w-d3d12-debug", p + 1)) {
        const std::size_t end = log.find('\n', p);
        const std::string line = log.substr(p, end == std::string::npos ? std::string::npos : end - p);
        std::printf("%s\n", line.c_str());
        if (line.find("sev=0 ") != std::string::npos || line.find("sev=1 ") != std::string::npos) ++failures;
    }
    std::printf("device removed reason: 0x%08lx\n", static_cast<unsigned long>(desktop->GetDeviceRemovedReason()));
    if (desktop->GetDeviceRemovedReason() != S_OK) ++failures;
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 16 : 0;
}
