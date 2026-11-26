#include "xmem_test_device.h"
#include "desktop_gpu.h"
#include "xmem_section_lease.h"
#include "xmem_page_batch.h"
#include <thread>

#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) void* WINAPI XMemVirtualAlloc(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
extern "C" __declspec(dllimport) BOOL WINAPI XMemAllocatePhysicalPages(ULONG_PTR, ULONG_PTR*, ULONG_PTR*);
extern "C" __declspec(dllimport) BOOL WINAPI XMemFreePhysicalPages(ULONG_PTR, ULONG_PTR*);
extern "C" __declspec(dllimport) void* WINAPI XMemMapPhysicalPages(void*, ULONG_PTR, const ULONG_PTR*);
extern "C" __declspec(dllimport) BOOL WINAPI XMemMapPhysicalPagesScatter(void* const*, ULONG_PTR, const ULONG_PTR*);
extern "C" __declspec(dllimport) ULONG64 WINAPI x2w_xmem_gpu_mapping_generation();
extern "C" __declspec(dllimport) BOOL WINAPI x2w_xmem_gpu_location(const void*, HANDLE*, ULONG64*);
extern "C" __declspec(dllimport) BOOL WINAPI x2w_xmem_lease_section(UINT, ULONG_PTR, SIZE_T, X2WXMemSectionLease*, SIZE_T);
extern "C" __declspec(dllimport) BOOL WINAPI x2w_xmem_release_section(UINT, ULONG64);
extern "C" __declspec(dllimport) UINT64 x2w_gpu_find_address_test(UINT64, UINT64);
extern "C" __declspec(dllimport) BOOL x2w_desktop_range_test(UINT64, UINT64, ID3D12Resource**, UINT64*);

using Microsoft::WRL::ComPtr;
constexpr SIZE_T page = 65536, mib = 1 << 20;

void require(bool ok, const char* what)
{
    if (!ok) throw std::runtime_error(what);
}

HRESULT create_wrapper(IUnknown* adapter, REFIID iid, void** out)
{
    unsigned char params[64]{};
    return D3D12XboxCreateDevice(adapter, params, iid, out);
}

int main(int argc, char** argv) try
{
    if (argc < 2 || argc > 3) return 2;
    const bool hardware = argc == 3 && !std::strcmp(argv[2], "hardware");
    if (argc == 3 && !hardware) return 2;
    const bool linear = std::strcmp(argv[1], "linear") == 0;
    if (!linear && std::strcmp(argv[1], "baseline")) return 2;
    SetEnvironmentVariableA("X2W_XMEM_PERMISSIVE", "1");
    SetEnvironmentVariableA("X2W_D3D12_PERMISSIVE", "1");
    SetEnvironmentVariableA("X2W_XMEM_LINEAR_PLACEMENT", linear ? "1" : "0");
    SetEnvironmentVariableA("X2W_XMEM_GPU_POOL", "0");
    SetEnvironmentVariableA("X2W_XMEM_SLAB_DIR", nullptr);
    const std::array<SIZE_T, 6> boundaries{0, 10, 11, 38, 85, 185};
    auto* root = static_cast<unsigned char*>(XMemVirtualAlloc(nullptr, 185 * mib, 0x42000, 7, PAGE_NOACCESS));
    require(root != nullptr, "reserve parent");
    for (unsigned i = 0; i < 5; ++i) {
        auto* child = root + boundaries[i] * mib;
        const SIZE_T bytes = (boundaries[i + 1] - boundaries[i]) * mib;
        if (i < 4) require(VirtualFree(child, bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER), "split child");
        require(XMemVirtualAlloc(child, bytes, 0x20406000, 0x27, PAGE_READWRITE) == child, "register child");
    }
    ULONG_PTR count = 185 * mib / page;
    std::vector<ULONG_PTR> tokens(count);
    require(XMemAllocatePhysicalPages(0x280, &count, tokens.data()) && count == tokens.size(), "initial grant");
    for (unsigned i : {3u, 4u, 0u, 1u, 2u}) {
        void* child = root + boundaries[i] * mib;
        const ULONG_PTR pages = (boundaries[i + 1] - boundaries[i]) * mib / page;
        require(XMemMapPhysicalPages(child, pages, tokens.data() + boundaries[i] * mib / page) == child, "map child");
    }
    auto* start = root + 11 * mib;
    HANDLE section{};
    ULONG64 base_at{};
    require(x2w_xmem_gpu_location(start, &section, &base_at), "arena location");
    unsigned mismatches = 0;
    for (SIZE_T at = 0; at < 174 * mib; at += page) {
        HANDLE h{};
        ULONG64 offset{};
        require(x2w_xmem_gpu_location(start + at, &h, &offset), "page location");
        if (h != section || offset != base_at + at) ++mismatches;
    }
    require((mismatches == 0) == linear, "physical continuity control");
    std::printf("physical pages %zu, noncontiguous %u\n", 174 * mib / page, mismatches);
    bool unmatched = false;
    auto wrapper = create_device(!hardware, unmatched, create_wrapper);
    if (hardware && !wrapper) return 77;
    require(wrapper.Get() != nullptr, "create device wrapper");
    ComPtr<ID3D12Resource> window;
    UINT64 at{};
    const BOOL bound = x2w_desktop_range_test(reinterpret_cast<UINT64>(start), 174 * mib, window.GetAddressOf(), &at);
    require(!!bound == linear, "full 174 MiB desktop_range control");
    if (bound) {
        ComPtr<ID3D12Device> device;
        check(window->GetDevice(IID_PPV_ARGS(&device)), "window device");
        const auto readback = make_buffer(device.Get(), 16, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        ComputeCommands commands(device.Get(), nullptr);
        const SIZE_T offsets[]{0, 27 * mib, 74 * mib, 174 * mib - 4};
        for (unsigned j = 0; j < 4; ++j) {
            *reinterpret_cast<UINT*>(start + offsets[j]) = 100 + j;
            commands.list->CopyBufferRegion(readback.Get(), j * 4, window.Get(), at + offsets[j], 4);
        }
        commands.submit_and_wait(device.Get());
        UINT* actual = nullptr;
        check(readback->Map(0, nullptr, reinterpret_cast<void**>(&actual)), "read GPU results");
        for (unsigned j = 0; j < 4; ++j) require(actual[j] == 100 + j, "GPU readback across child boundaries");
        readback->Unmap(0, nullptr);
    }
    ULONG_PTR token = tokens[11 * mib / page], hole = 0;
    void* address = start;
    void* alias = XMemVirtualAlloc(nullptr, 2 * page, 0x20402000, 0x21, PAGE_READWRITE);
    require(alias != nullptr, "reserve alias");
    const auto address_of = [](void* pointer) -> UINT64 {
        ComPtr<ID3D12Resource> buffer;
        UINT64 offset{};
        return x2w_desktop_range_test(reinterpret_cast<UINT64>(pointer), 4, buffer.GetAddressOf(), &offset)
            ? buffer->GetGPUVirtualAddress() + offset : 0;
    };
    // A fresh thread cannot have a TLS hit: it consults the shared cache before
    // the original thread is allowed to replace either result.
    const auto verify_caches = [&](UINT64 expected) {
        UINT64 fresh{};
        std::thread lookup([&] { fresh = address_of(alias); });
        lookup.join();
        require(fresh == expected && address_of(alias) == expected, "current GPU address on both cache paths");
        require(x2w_gpu_find_address_test(reinterpret_cast<UINT64>(alias), 4) == expected, "translated buffer binding tracks remaps");
    };
    require(address_of(alias) == 0, "cache negative before first mapping");
    require(XMemMapPhysicalPagesScatter(&alias, 1, &token), "map original grant alias");
    const UINT64 original = address_of(start);
    const UINT64 other = address_of(start + page);
    require(original && other && original != other, "two distinct GPU backing addresses");
    verify_caches(original);
    X2WPageRange range{0, 1, 1};
    X2WPageBatch batch{reinterpret_cast<UINT64>(alias), 1, &range};
    X2WPageBatchStats stats{};
    require(SUCCEEDED(X2WMapPageBatch(1, &batch, reinterpret_cast<UINT64>(start), 2, 0, &stats)) && stats.pagesChanged == 1,
            "batch remap alias to next page");
    verify_caches(other);
    const auto before = x2w_xmem_gpu_mapping_generation();
    require(SUCCEEDED(X2WMapPageBatch(1, &batch, reinterpret_cast<UINT64>(start), 2, 0, &stats)) &&
            !stats.pagesChanged && x2w_xmem_gpu_mapping_generation() == before, "unchanged batch keeps the cache warm");
    // The first edit can be installed. The second page is inside the same reservation
    // but is now privately committed, forcing the OS placement to fail and roll back.
    auto* occupied = static_cast<unsigned char*>(alias) + page;
    require(VirtualAlloc2(GetCurrentProcess(), occupied, page, MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                          PAGE_READWRITE, nullptr, 0) == occupied, "occupy rollback control page");
    X2WPageRange move{1, 0, 2};
    X2WPageBatch rejected{reinterpret_cast<UINT64>(alias), 1, &move};
    require(FAILED(X2WMapPageBatch(1, &rejected, reinterpret_cast<UINT64>(start), 2, 0, &stats)) &&
            x2w_xmem_gpu_mapping_generation() > before, "failed native transaction invalidates cached bindings");
    verify_caches(other);
    require(XMemMapPhysicalPages(alias, 1, nullptr) == alias, "unmap alias with cached hit");
    verify_caches(0);
    require(XMemMapPhysicalPages(alias, 1, &token) == alias, "map original grant again");
    verify_caches(original);
    *static_cast<UINT*>(alias) = 0x12345678;
    require(*reinterpret_cast<UINT*>(start) == 0x12345678, "coherent CPU alias");
    X2WXMemSectionLease lease{};
    constexpr UINT version = x2w_xmem_single_page_lease_version;
    require(x2w_xmem_lease_section(version, reinterpret_cast<ULONG_PTR>(start), 4, &lease, sizeof(lease)), "pin old slot");
    auto* old = static_cast<UINT*>(MapViewOfFile(lease.section, FILE_MAP_READ, 0, static_cast<DWORD>(lease.section_offset), page));
    require(old != nullptr, "leased bytes");
    require(XMemMapPhysicalPagesScatter(&alias, 1, &hole), "unmap alias");
    require(XMemMapPhysicalPages(root, tokens.size(), nullptr) == root, "unmap complete grant");
    require(XMemFreePhysicalPages(tokens.size(), tokens.data()), "retire complete grant");
    count = 1;
    require(XMemAllocatePhysicalPages(0x280, &count, &token) && XMemMapPhysicalPagesScatter(&address, 1, &token), "replacement grant");
    require(*reinterpret_cast<UINT*>(start) == 0, "replacement zero");
    *reinterpret_cast<UINT*>(start) = 0x87654321;
    require(*old == 0x12345678, "retired slot unchanged");
    require(UnmapViewOfFile(old) && CloseHandle(lease.section) && x2w_xmem_release_section(version, lease.id), "release retired lease");
    require(XMemMapPhysicalPagesScatter(&address, 1, &hole) && XMemFreePhysicalPages(1, &token)
            && XMemAllocatePhysicalPages(0x280, &count, &token) && XMemMapPhysicalPagesScatter(&address, 1, &token), "reuse grant");
    HANDLE reused{};
    ULONG64 reused_at{};
    require(x2w_xmem_gpu_location(start, &reused, &reused_at) && (!linear || (reused == section && reused_at == base_at)) &&
            *reinterpret_cast<UINT*>(start) == 0, "replacement zeroed and preferred slot reusable after lease release");
    std::printf("desktop_range %s; alias, pin, retirement, reuse passed\n", bound ? "bound; GPU readback 4/4" : "NULL");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAILED: %s, Win32 %lu\n", error.what(), GetLastError());
    return 1;
}
