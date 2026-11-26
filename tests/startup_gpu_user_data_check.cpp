// X2W_GPU_ABI (WP3 vbind): the user-data dwords translated DXIL reads at b0 space0 (gpu_roots.h, a port of
// vg/draw.rs user_value / host_word / bind :1527-1551), the fresh-table planning (draw.rs tables), raw-view bounding
// (root_srv_table / root_uav), the memory adapter's bounds (find_bounded), RecordHash and the mesh clamp.
//
// The ABI (the upstream emitter, src/emit/mod.rs): a table's dword is its byte offset in its heap in CONSOLE units, 32 bytes a
// view and 16 a sampler, whatever the desktop increment is.
//
// --mutate is the negative control: table offsets computed as index x a desktop increment (64), the bias found again
// from the address instead of taken from the bound view, and gpu_find bounded by the window alone. It must fail.
#include "startup_stubs/gpu_roots.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>

namespace {
int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
            ++failures;                                                          \
        }                                                                        \
    } while (0)
bool mutate = false;
using namespace x2w_vbind;
using x2w_gpu::GpuRootIndexedView;
using x2w_gpu::GpuUsedResource;

GpuUsedResource use(GpuResourceType kind, GpuRootType root, std::uint32_t offset_dwords, std::uint8_t slot)
{
    GpuUsedResource u{};
    u.kind = kind;
    u.root = root;
    u.table_offset_dwords = offset_dwords;
    u.dwords = 8;
    u.slot = slot;
    return u;
}

std::vector<std::uint32_t> fill(const Roots& r, bool compute, const std::vector<GpuUserData>& words, std::uint32_t dwords,
                                const std::vector<MadeTable>& made, const HeapSpan* heaps, const std::vector<BoundBias>& biases,
                                unsigned* misses = nullptr)
{
    auto v = fill_user_data(r, compute, words.data(), words.size(), dwords, made, heaps, 2, biases, 0x77,
        [](std::uint32_t slot, std::uint32_t dword) -> std::uint64_t { return slot == 4 ? 0x1234500000ull + dword * 4 : 0; }, misses);
    if (mutate) {
        // The desktop-increment mistake: a table's dword as index x 64.
        for (const GpuUserData& u : words)
            if (u.root == GpuRootType::table && u.index < dwords)
                if (const GpuRoot* p = r.at(compute, u.slot); p && p->kind == GpuRootKind::table)
                    if (const HeapSpan* h = heap_of(heaps, 2, p->handle))
                        v[u.index] = static_cast<std::uint32_t>((p->handle - h->start) / h->step() * 64);
        // Recover the bias from the address when decoding the bound view.
        for (const GpuUserData& u : words)
            if (u.root == GpuRootType::other && u.other == abi::user_view_bias && u.index < dwords)
                if (const GpuRoot* p = r.at(compute, u.slot); p && is_view(p->kind)) v[u.index] = static_cast<std::uint32_t>(p->address & 15);
    }
    return v;
}

void user_data()
{
    HeapSpan heaps[2] = {{0x10000000, 1000, false}, {0x20000000, 100, true}};
    Roots r;
    r.set_table(false, 0, 0x10000000 + 5 * 32);   // view table at descriptor 5
    r.set_table(false, 1, 0x20000000 + 3 * 16);   // sampler table at descriptor 3
    const std::uint32_t c[6] = {10, 11, 12, 13, 14, 15};
    r.set_constants(false, 2, c, 6, 0);
    r.set_view(false, 3, 0x0000004512345678ull, GpuRootKind::srv);  // a root view at an address ending in 8
    r.set_table(false, 4, 0x10000000 + 7 * 32);
    r.set_table(false, 6, 0x30000000);            // outside both heaps
    std::vector<GpuUserData> words = {
        {0, 0, GpuRootType::table, 0, 0},
        {1, 1, GpuRootType::table, 0, 0},
        {2, 2, GpuRootType::constants, 0, 4},
        {3, 3, GpuRootType::other, abi::user_root_address, 0},
        {4, 3, GpuRootType::other, abi::user_root_address + 1, 0},
        {5, 3, GpuRootType::other, abi::user_view_bias, 0},
        {6, 4, GpuRootType::other, abi::user_view_address, 8},
        {7, 4, GpuRootType::other, abi::user_view_address + 1, 8},
        {8, 0, GpuRootType::other, abi::user_dispatch_threads_x, 64},
        {9, 0, GpuRootType::other, abi::user_memory_base, 0},
        {10, 6, GpuRootType::table, 0, 0},     // outside the heaps: 0, a miss
        {11, 9, GpuRootType::constants, 0, 0}, // never set: 0, a miss
        {12, 2, GpuRootType::constants, 0, 9}, // past the constants set: 0
        {13, 2, GpuRootType::table, 0, 0},     // the root is constants, the shader reads a table: 0, a miss
        {20, 2, GpuRootType::constants, 0, 0}, // past `dwords`: left out
    };
    // The bias as bound: the raw view was made from the 16 bytes below a window offset ending in 8.
    const std::vector<BoundBias> biases = {{3, 8}};
    unsigned misses = 0;
    const auto v = fill(r, false, words, 16, {}, heaps, biases, &misses);
    CHECK(v.size() == 16);
    CHECK(v[0] == 5 * 32);        // console bytes, not index x desktop increment
    CHECK(v[1] == 3 * 16);        // samplers: 16 a descriptor
    CHECK(v[2] == 14);
    CHECK(v[3] == 0x12345678u && v[4] == 0x45u);
    CHECK(v[5] == 8);
    CHECK(v[6] == 0x34500020u && v[7] == 0x12u);
    CHECK(v[8] == 0);             // all threads of every group
    CHECK(v[9] == 0x77);
    CHECK(v[10] == 0 && v[11] == 0 && v[12] == 0 && v[13] == 0);
    CHECK(misses == 3);
    // The bias is the bound view's, not the address's: a view bound through a copy whose offset ends in 4.
    const auto w = fill(r, false, words, 16, {}, heaps, {{3, 4}});
    CHECK(w[5] == 4);
    // A table made afresh gives its new offset.
    const auto f = fill(r, false, words, 16, {{0, 4096 * 32}}, heaps, biases);
    CHECK(f[0] == 4096 * 32 && f[1] == 3 * 16);
    // Compute roots are not graphics roots.
    const auto g = fill(r, true, words, 16, {}, heaps, biases);
    CHECK(g[0] == 0 && g[2] == 0);
}

void tables()
{
    HeapSpan heaps[2] = {{0x10000000, 1000, false}, {0x20000000, 100, true}};
    Roots r;
    r.set_table(false, 0, 0x10000000 + 10 * 32);  // a table with a CBV: fresh
    r.set_table(false, 1, 0x10000000 + 990 * 32); // reaches past 999 of the host's 995: fresh
    r.set_table(false, 2, 0x10000000 + 20 * 32);  // SRVs only: mirrored in place
    r.set_table(false, 3, 0x20000000);            // a sampler table with a CBV entry: skipped
    const std::uint32_t c = 3;
    r.set_constants(false, 5, &c, 1, 0);
    std::vector<GpuUsedResource> used = {
        use(GpuResourceType::cbv, GpuRootType::table, 0, 0),
        use(GpuResourceType::srv, GpuRootType::table, 2 * 8, 0),
        use(GpuResourceType::srv, GpuRootType::table, 6 * 8, 1),
        use(GpuResourceType::srv, GpuRootType::table, 1 * 8, 2),
        use(GpuResourceType::uav, GpuRootType::table, 0, 2),
        use(GpuResourceType::cbv, GpuRootType::table, 0, 3),
        use(GpuResourceType::srv, GpuRootType::descriptor, 0, 4),  // a root SRV: no table
    };
    // A descriptor root state selects: byte offset 32 + (3 << 5) = 128 -> entry 4 of table 0.
    std::vector<GpuRootIndexedView> indexed = {{0, 5, 0, 32, 5}};
    const auto plan = plan_tables(r, false, used.data(), used.size(), indexed.data(), indexed.size(), heaps, 2,
                                  [](const HeapSpan&) { return 995u; });
    CHECK(plan.fresh.size() == 2);
    if (plan.fresh.size() == 2) {
        CHECK(plan.fresh[0].slot == 0 && plan.fresh[0].n == 5 && plan.fresh[0].selected == std::vector<std::uint32_t>{4});
        CHECK(plan.fresh[1].slot == 1 && plan.fresh[1].n == 7);
    }
    CHECK(plan.mirrored == (std::vector<std::size_t>{3, 4}));
    CHECK(plan.skipped.size() == 1 && plan.skipped[0].first == 3 && plan.skipped[0].second == PlanSkip::sampler_or_unaligned);
    // Keep the title table when the requested span extends past the heap.
    Roots distant;
    distant.set_table(false, 0, 0x10000000 + 998 * 32);
    std::vector<GpuUsedResource> long_cbv = {use(GpuResourceType::cbv, GpuRootType::table, 4 * 8, 0)};
    const auto p2 = plan_tables(distant, false, long_cbv.data(), 1, nullptr, 0, heaps, 2, [](const HeapSpan& h) { return h.count; });
    CHECK(p2.fresh.empty() && p2.skipped.size() == 1 && p2.skipped[0].second == PlanSkip::too_long);
}

void raw_views()
{
    RawWindow w;
    CHECK(raw_window(0x1008, 100, true, &w) && w.first == 0x1000 && w.bytes == 108 && w.bias == 8);
    CHECK(!raw_window(0x1008, 100, false, &w));  // unbiased and not on 16: null ("reads zeros")
    CHECK(raw_window(0x1010, 100, false, &w) && w.first == 0x1010 && w.bytes == 100 && w.bias == 0);
    CHECK(!raw_window(0x1010, 0, true, &w));
    CHECK(raw_elements(2) == 1 && raw_elements(10) == 2 && raw_elements(1ull << 40) == 0xffffffffu);
    CHECK(constant_view_bytes(16) == 256 && constant_view_bytes(257) == 512 && constant_view_bytes(1 << 20) == 65536);
    CHECK(root_cbv_bytes(1 << 20) == 65536 && root_cbv_bytes(100) == 100);
}

// find_bounded: a title buffer [0x100000, 0x140000) inside a window [0x0, 0x1000000) whose desktop pages run on
// contiguously only up to 0x120000 (the window is cut there).
void find()
{
    const auto extent = [](std::uint64_t a, std::uint64_t* base, std::uint64_t* end) {
        if (a < 0x100000 || a >= 0x140000) return false;
        *base = 0x100000;
        *end = 0x140000;
        return true;
    };
    static int window_token;
    const auto window = [](std::uint64_t a, std::uint64_t n, void** buffer, std::uint64_t* offset) {
        if (a + n > 0x1000000) return false;
        if (!mutate && a < 0x120000 && a + n > 0x120000) return false;  // mutate: the window "allows" everything
        *buffer = &window_token;
        *offset = a;
        return true;
    };
    const auto bounded = [&](std::uint64_t a, std::uint64_t* base, std::uint64_t* end) {
        if (mutate) {  // mutate: no title buffer bound, the window's end instead
            *base = 0;
            *end = 0x1000000;
            return a < 0x1000000;
        }
        return extent(a, base, end);
    };
    TitleBufferExtent inner, span;
    CHECK(inner.consider(0x1800, 1, 0x1000, 0x8000, 1));
    CHECK(inner.consider(0x1800, 1, 0x1800, 0x100, 2));
    CHECK(!inner.consider(0x1800, 1, 0x1000, 0x8000, 1) && inner.end == 0x1900);
    CHECK(span.consider(0x1800, 0x200, 0x1800, 0x100, 2));
    CHECK(span.consider(0x1800, 0x200, 0x1000, 0x8000, 1) && span.end == 0x9000);
    Found f;
    // Inside the buffer, before the cut: bounded by the cut (0x120000), not the window end.
    CHECK(find_bounded(0x110000, 1, ~0ull, 0, bounded, window, &f) && f.known_extent && f.left == 0x10000 && f.cut_by_window);
    // After the cut: bounded by the buffer's end.
    CHECK(find_bounded(0x130000, 1, ~0ull, 0, bounded, window, &f) && f.left == 0x10000 && !f.cut_by_window);
    // A cap (X2W_SHADERCFG_ROOT_UAV_MB) applies.
    CHECK(find_bounded(0x130000, 1, 0x100, 0, bounded, window, &f) && f.left == 0x100);
    // Memory no title buffer holds: refused without an allowance, bounded by it with one.
    CHECK(!find_bounded(0x500000, 1, ~0ull, 0, bounded, window, &f) && !f.known_extent);
    CHECK(find_bounded(0x500000, 1, ~0ull, 0x4000, bounded, window, &f) && f.left == 0x4000);
    // More than is left: refused.
    CHECK(!find_bounded(0x13ff00, 0x200, ~0ull, 0, bounded, window, &f));
}

void hashes()
{
    const auto of = [](const std::vector<std::array<std::uint32_t, 3>>& records) {
        RecordHash h;
        for (const auto& r : records) h.add(r[0], r[1], r[2]);
        return h.value;
    };
    std::set<std::uint64_t> seen;
    bool distinct = true;
    for (std::uint32_t a = 0; a < 64; ++a)
        for (std::uint32_t b = 0; b < 64; ++b)
            for (std::uint32_t c : {0u, 1u, 0x80000000u, 0xffffffffu}) {
                distinct &= seen.insert(of({{a, b, c}})).second;
                distinct &= seen.insert(of({{1, 2, 3}, {a, b, c}})).second;
                distinct &= seen.insert(of({{a, b, c}, {1, 2, 3}, {0, 0, 0}})).second;
            }
    distinct &= seen.insert(of({})).second;
    for (int n = 2; n < 100; ++n) distinct &= seen.insert(of(std::vector<std::array<std::uint32_t, 3>>(n, {0, 0, 0}))).second;
    CHECK(distinct);  // draw.rs records_that_differ_hash_apart
    std::uint32_t record[8]{};
    RecordHash original;
    original.add_record(record);
    for (auto& word : record) {
        word = 1;
        RecordHash changed;
        changed.add_record(record);
        CHECK(original.value != changed.value);
        word = 0;
    }
}

void dispatch()
{
    const std::uint32_t size[3] = {64, 4, 4};
    std::uint32_t out[3];
    const std::uint32_t t1[3] = {8847, 1, 1};
    dispatch_groups(t1, size, out);
    CHECK(out[0] == 139 && out[1] == 1 && out[2] == 1);
    const std::uint32_t t2[3] = {0, 12, 9};
    dispatch_groups(t2, size, out);
    CHECK(out[0] == 0 && out[1] == 3 && out[2] == 3);
    CHECK(mesh_groups(70000) == 65535 && mesh_groups(12) == 12);
}
}  // namespace

int main(int argc, char** argv)
{
    mutate = argc > 1 && std::strcmp(argv[1], "--mutate") == 0;
    user_data();
    tables();
    raw_views();
    find();
    hashes();
    dispatch();
    std::printf("%s: %d failure(s)\n", mutate ? "gpu user data (mutated control)" : "gpu user data", failures);
    return failures ? 1 : 0;
}
