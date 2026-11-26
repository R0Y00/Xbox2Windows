#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
using Word = std::uint64_t;
template <class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }
void** table_of(void* object) { return *static_cast<void***>(object); }
Word invoke(void* object, unsigned index, Word a = 0, Word b = 0, Word c = 0, Word d = 0, Word e = 0)
{
    using Slot = Word(WINAPI*)(void*, Word, Word, Word, Word, Word);
    return reinterpret_cast<Slot>(table_of(object)[index])(object, a, b, c, d, e);
}
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}
struct TitleHeapDesc {
    D3D12_DESCRIPTOR_HEAP_DESC desc;
    std::uint64_t extra[2];
};
}  // namespace

int main()
{
    SetEnvironmentVariableA("X2W_GPU_ABI", "1");
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter> adapter;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) return 3;
    unsigned char parameters[64]{};
    void* device = nullptr;
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters, __uuidof(ID3D12Device), &device))) return 4;

    // The title's order: a shader-visible view heap over title memory (32-byte records under ABI), then a 128-slot
    // shader-visible sampler heap with no title memory (extra = 0, 0x800).
    constexpr UINT views = 1024, view_step = 32, samplers = 128, sampler_step = 16;
    std::vector<unsigned char> title(views * view_step + 0x10000);
    const std::uint64_t title_cpu = (reinterpret_cast<std::uint64_t>(title.data()) + 0xffff) & ~std::uint64_t{0xffff};
    TitleHeapDesc view_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, views, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 1},
                            {title_cpu, std::uint64_t{views} * view_step}};
    TitleHeapDesc sampler_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, samplers, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 1},
                               {0, std::uint64_t{samplers} * sampler_step}};
    void* heaps[2] = {};
    const TitleHeapDesc* descs[2] = {&view_desc, &sampler_desc};
    const char* names[2] = {"view heap", "sampler heap"};
    for (int i = 0; i < 2; ++i)
        if (FAILED(static_cast<HRESULT>(invoke(device, 14, arg(descs[i]), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&heaps[i])))) ||
            !heaps[i])
            return 5 + i;

    for (int i = 0; i < 2; ++i) {
        void* heap = heaps[i];
        void** const table = table_of(heap);
        char what[160];
        // A clean call (second argument 0) gives the expected answers.
        const Word cpu = invoke(heap, 9);
        const Word gpu = invoke(heap, 10);
        std::printf("%s %p: cpu start 0x%llx, gpu start 0x%llx\n", names[i], heap, static_cast<unsigned long long>(cpu),
                    static_cast<unsigned long long>(gpu));
        if (i == 0) {
            std::snprintf(what, sizeof(what), "%s: slot 9 returns the title memory", names[i]);
            check(cpu == title_cpu, what);
        }
        // The title's call: the stale second argument is a wrapper (this heap, the other heap, the device).
        void* stale[3] = {heap, heaps[1 - i], device};
        for (void* s : stale) {
            void** const stale_table = table_of(s);
            const Word cpu_stale = invoke(heap, 9, arg(s));
            std::snprintf(what, sizeof(what), "%s: slot 9 with a stale wrapper %p in rdx keeps both table pointers", names[i], s);
            check(table_of(heap) == table && table_of(s) == stale_table, what);
            if (table_of(heap) != table || table_of(s) != stale_table) return 1;  // a call through it would execute data
            std::snprintf(what, sizeof(what), "%s: slot 9 with a stale wrapper in rdx returns the start (0x%llx), not the object",
                          names[i], static_cast<unsigned long long>(cpu_stale));
            check(cpu_stale == cpu, what);
            std::snprintf(what, sizeof(what), "%s: slot 10 is not null", names[i]);
            check(table_of(heap)[10] != nullptr, what);
            const Word gpu_stale = invoke(heap, 10, arg(s));
            std::snprintf(what, sizeof(what), "%s: slot 10 with a stale wrapper in rdx returns the GPU start (0x%llx)", names[i],
                          static_cast<unsigned long long>(gpu_stale));
            check(gpu_stale == gpu && gpu != 0 && table_of(heap) == table, what);
        }
    }
    // The title stages a console sampler on the CPU, then copies it to the GPU-visible
    // heap, both with no supplied address. Both starts must expose plain 16-byte records.
    sampler_desc.desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    void* staging = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(device, 14, arg(&sampler_desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&staging)))) ||
        !staging) return 7;
    const Word stage = invoke(staging, 9);
    const Word visible = invoke(heaps[1], 9);
    for (const Word start : {stage, visible}) {
        MEMORY_BASIC_INFORMATION mem{};
        if (!VirtualQuery(reinterpret_cast<void*>(start), &mem, sizeof(mem)) || mem.State != MEM_COMMIT ||
            mem.Protect != PAGE_READWRITE || arg(mem.AllocationBase) != start) {
            std::puts("FAIL unplaced sampler heap has no plain title memory");
            return 1;
        }
    }
    constexpr Word from = 4 * sampler_step, to = 9 * sampler_step;
    std::uint32_t words[27]{};
    words[1] = words[2] = 1;  // linear min and mag
    invoke(device, 61, arg(words), stage + from);
    std::uint32_t record[4]{};
    std::memcpy(record, reinterpret_cast<const void*>(stage + from), sizeof(record));
    check(record[0] == 0x56573258 && record[2] == 22, "staged console sampler became a 16-byte sampler record");
    invoke(device, 24, 1, visible + to, stage + from, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    check(std::memcmp(record, reinterpret_cast<const void*>(visible + to), sizeof(record)) == 0,
          "CopyDescriptorsSimple preserved the staged sampler at its visible index");
    void* allocator = nullptr;
    void* list = nullptr;
    const unsigned list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    if (FAILED(static_cast<HRESULT>(invoke(device, 9, 0, arg(&__uuidof(ID3D12CommandAllocator)), arg(&allocator)))) || !allocator ||
        FAILED(static_cast<HRESULT>(invoke(device, 73, arg(list_desc), arg(allocator), 0, arg(&__uuidof(ID3D12GraphicsCommandList)), arg(&list)))) ||
        !list || FAILED(static_cast<HRESULT>(invoke(list, 139, arg(allocator))))) return 8;
    invoke(list, 32, 2, arg(heaps));
    if (FAILED(static_cast<HRESULT>(invoke(list, 9)))) return 9;
    if (failures) std::printf("startup_gpu_abi_heap_start: %d failures\n", failures);
    else std::printf("startup_gpu_abi_heap_start: pass\n");
    return failures ? 1 : 0;
}
