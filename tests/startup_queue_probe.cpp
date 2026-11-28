// Unknown console queue slots must not fault on every call (startup_queue_probe).
//
//   startup_queue_probe <warp|hardware> reserved
//   startup_queue_probe <warp|hardware> xmem <xmem.dll>
//
// The title called queue slot 31 every frame with r9 = the base of a physical-page pool whose mapping
// had failed: reserved memory with nothing behind it. Slot 31 now maps pages (queue_page_mappings), but
// any slot still left to the shape dispatch (queue_dynamic) checks pointer-shaped arguments for an IID,
// and the check touched them on every call: one first-chance access violation per call (423 in one run
// at d3d12_xs probe_readable). This test calls slot 30, which has no known meaning, the same way.
//
//   reserved: the address is a plain VirtualAlloc reservation (unknown to xmem). 200 calls may fault at
//             most 32 times (the first calls of a slot are still checked by probing).
//   xmem:     the xmem shim is loaded as xmem.dll and the address is an XMemVirtualAlloc reservation with
//             no pages mapped. xmem's page registry answers for it, so 200 calls fault 0 times.
//
// Every call must return 0 (the slot stays a no-op). A vectored handler counts access violations at the
// address; it lets them continue to the stand-in's own handler.
// Exit codes: 0 pass, 77 no physical adapter, other values name the failing step.
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
using Word = std::uint64_t;
template <class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }
Word invoke(void* object, unsigned index, Word a = 0, Word b = 0, Word c = 0, Word d = 0)
{
    using Slot = Word (WINAPI*)(void*, Word, Word, Word, Word);
    return reinterpret_cast<Slot>((*static_cast<void***>(object))[index])(object, a, b, c, d);
}

volatile std::uintptr_t watched_base, watched_end;
volatile LONG faults;
LONG CALLBACK count_faults(EXCEPTION_POINTERS* e)
{
    const EXCEPTION_RECORD* r = e->ExceptionRecord;
    if (r->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && r->NumberParameters >= 2 &&
        r->ExceptionInformation[1] >= watched_base && r->ExceptionInformation[1] < watched_end)
        InterlockedIncrement(&faults);
    return EXCEPTION_CONTINUE_SEARCH;
}

using XMemVirtualAllocFn = void*(WINAPI*)(void*, SIZE_T, DWORD, ULONG64, DWORD);
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3) return 2;
    const bool use_xmem = std::strcmp(argv[2], "xmem") == 0;
    if (!use_xmem && std::strcmp(argv[2], "reserved") != 0) return 2;
    if (use_xmem && argc < 4) return 2;
    XMemVirtualAllocFn xmem_reserve = nullptr;
    if (use_xmem) {
        SetEnvironmentVariableW(L"X2W_XMEM_PERMISSIVE", L"1");
        const HMODULE xmem = LoadLibraryA(argv[3]);
        if (!xmem || GetModuleHandleW(L"xmem.dll") != xmem) {
            std::fprintf(stderr, "cannot load %s as xmem.dll\n", argv[3]);
            return 20;
        }
        xmem_reserve = reinterpret_cast<XMemVirtualAllocFn>(GetProcAddress(xmem, "XMemVirtualAlloc"));
        if (!xmem_reserve || !GetProcAddress(xmem, "x2w_xmem_mapped_bytes")) return 21;
    }

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

    std::array<unsigned char, 64> parameters{};
    void* device = nullptr;
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device), &device))) return 4;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    void* queue = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(device, 8, arg(&queue_desc), arg(&__uuidof(ID3D12CommandQueue)), arg(&queue)))))
        return 5;

    constexpr SIZE_T bytes = 16 * 0x10000;
    void* base = use_xmem ? xmem_reserve(nullptr, bytes, 0x20402000, 0x23, PAGE_READWRITE)
                          : VirtualAlloc(nullptr, bytes, MEM_RESERVE, PAGE_NOACCESS);
    if (!base) return 6;
    watched_base = reinterpret_cast<std::uintptr_t>(base);
    watched_end = watched_base + bytes;
    void* handler = AddVectoredExceptionHandler(1, count_faults);
    if (!handler) return 7;

    // The title's shape: a small count, a stack array, then the pool base.
    std::array<Word, 6> batch{0x2, 0x200000003, 0x300000000, 0x2, 0x200000003, 0x300000000};
    constexpr int calls = 200;
    constexpr unsigned unknown_slot = 30;
    for (int i = 0; i < calls; ++i) {
        const Word result = invoke(queue, unknown_slot, 1 + (i & 1) * 2, arg(batch.data()), arg(base));
        if (result != 0) {
            std::printf("call %d returned 0x%llx\n", i, static_cast<unsigned long long>(result));
            return 8;
        }
    }
    RemoveVectoredExceptionHandler(handler);
    const LONG seen = faults;
    const LONG allowed = use_xmem ? 0 : 32;
    std::printf("%s: %d calls of queue slot %u with r9=%p -> %ld access violations at the address (allowed %ld)\n",
                use_xmem ? "xmem" : "reserved", calls, unknown_slot, base, seen, allowed);
    return seen <= allowed ? 0 : 9;
}
