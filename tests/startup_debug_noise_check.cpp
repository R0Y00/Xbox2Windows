// Pure checks of the debug-noise rules: Map counts, optimized clear values, heap repairs, drain budget.
#include "startup_stubs/optimized_clear_value.h"
#include "startup_stubs/debug_drain_budget.h"
#include "startup_stubs/descriptor_heap_rules.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <initializer_list>
#include <cstring>
#include <limits>
#include <cstdlib>
#include <new>
using namespace x2w_startup;
// Allocation failure on demand, for the best-effort drain budget.
static bool fail_allocation = false;
void* operator new(std::size_t size) {
    if (fail_allocation) throw std::bad_alloc();
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
int main() {
    std::uint64_t maps = 0; assert(!consume_map_count(maps));
    maps = 2; assert(consume_map_count(maps) && maps == 1);
    assert(consume_map_count(maps) && !consume_map_count(maps));
    ResourceMapCounts resource_maps;
    int released = 0, live = 0;
    resource_maps.record(&released, 0);
    resource_maps.record(&released, 0);
    resource_maps.record(&released, 1);
    resource_maps.record(&live, 0);
    assert(resource_maps.consume(&released, 0));
    resource_maps.retire(&released);
    assert(!resource_maps.consume(&released, 0) && !resource_maps.consume(&released, 1));
    assert(resource_maps.consume(&live, 0) && !resource_maps.consume(&live, 0));
    resource_maps.record(&released, 0);
    assert(resource_maps.consume(&released, 0) && !resource_maps.consume(&released, 0));
    D3D12_RESOURCE_DESC d{}; D3D12_CLEAR_VALUE c{}, copy{};
    c.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    assert(!optimized_clear_value(nullptr, d, copy));
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    assert(!optimized_clear_value(&c, d, copy));
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    assert(!optimized_clear_value(&c, d, copy));
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    const auto original = c;
    assert(optimized_clear_value(&c, d, copy) == &c && !std::memcmp(&c, &original, sizeof(c)));
    c.Format = DXGI_FORMAT_UNKNOWN; assert(!optimized_clear_value(&c, d, copy));
    c.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS; assert(!optimized_clear_value(&c, d, copy));
    c.Format = DXGI_FORMAT_D32_FLOAT; assert(!optimized_clear_value(&c, d, copy));
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    c.Format = DXGI_FORMAT_R8G8B8A8_UNORM; assert(!optimized_clear_value(&c, d, copy));
    c.Format = DXGI_FORMAT_D32_FLOAT;
    for (float depth : {2.0f, -1.0f, (std::numeric_limits<float>::quiet_NaN)()}) {
        c.DepthStencil.Depth = depth;
        auto result = optimized_clear_value(&c, d, copy);
        assert(result == &copy && result->DepthStencil.Depth == (depth == 2.0f ? 1.0f : 0.0f));
    }
    c.DepthStencil.Depth = 0.5f; const auto valid_ds = c;
    assert(optimized_clear_value(&c, d, copy) == &c && !std::memcmp(&c, &valid_ds, sizeof(c)));
    D3D12_DESCRIPTOR_HEAP_DESC h{}; h.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    h.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE; h.NumDescriptors = 1000000;
    auto safe = safe_heap_desc(h); assert(!std::memcmp(&h, &safe, sizeof(h)));
    h.NumDescriptors++; assert(safe_heap_desc(h).NumDescriptors == 1000000);
    h.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE; h.NumDescriptors = 1048577;
    safe = safe_heap_desc(h); assert(!std::memcmp(&h, &safe, sizeof(h)));
    h.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER; h.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    h.NumDescriptors = 2048; safe = safe_heap_desc(h); assert(!std::memcmp(&h, &safe, sizeof(h)));
    h.NumDescriptors = 4096;
    assert(safe_heap_desc(h).NumDescriptors == 2048);
    for (auto type : {D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_DESCRIPTOR_HEAP_TYPE_DSV}) {
        h.Type = type; assert(safe_heap_desc(h).Flags == D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
    }
    using Decision = DebugDrainBudget::Decision;
    DebugDrainBudget b;
    for (unsigned i = 0; i < 500; ++i) b.record(2, i);
    assert(b.record(1, 1000) == Decision::message);
    DebugDrainBudget repeated; unsigned messages = 0, repeats = 0;
    for (unsigned i = 0; i < 100; ++i) { auto r = repeated.record(1, 7); messages += r == Decision::message; repeats += r == Decision::repeat; }
    assert(messages == 32 && repeats == 1 && repeated.count(7) == 100);
    DebugDrainBudget errors; messages = 0;
    for (unsigned i = 0; i < 5000; ++i) messages += errors.record(1, i) == Decision::message;
    assert(messages == 4000);
    // A view heap request up to 2^20 stays as asked on a tier 3 device (1000000+), is clamped to 1,000,000 elsewhere,
    // and a tier 3 device still refuses more than 2^20.
    h = {}; h.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; h.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    h.NumDescriptors = 1048576;
    assert(safe_heap_desc(h, max_tier3_view_descriptors).NumDescriptors == 1048576 && safe_heap_desc(h).NumDescriptors == 1000000);
    h.NumDescriptors = 1048577; assert(safe_heap_desc(h, max_tier3_view_descriptors).NumDescriptors == 1048576);
    // The floor is probed after the last halving step: a driver limit of 100,000 is found as 65,536.
    assert(largest_halving_size(1000000, 65536, [](std::uint64_t n) { return n <= 100000; }) == 65536);
    assert(largest_halving_size(1000000, 65536, [](std::uint64_t n) { return n <= 300000; }) == 250000);
    assert(largest_halving_size(1000000, 65536, [](std::uint64_t) { return false; }) == 0);
    assert(largest_halving_size(1000000, 65536, [](std::uint64_t n) { return n <= 65536; }) == 65536);
    // An allocation failure while counting a new message id drops the line instead of throwing.
    DebugDrainBudget fresh;
    fail_allocation = true;
    const auto dropped = fresh.record(1, 12345);
    fail_allocation = false;
    assert(dropped == Decision::suppress && fresh.count(12345) == 0);
    assert(fresh.record(1, 12345) == Decision::message);
}
