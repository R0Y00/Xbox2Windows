// Per-list graphics and compute root arguments, input-assembler views, topology, heaps and dynamic slots. User data
// is passed as root constants at b0 space0. A table dword is a byte offset in console heap units (32 bytes per
// view, 16 per sampler). RootType::Other codes select host-computed values.
#pragma once

#include "startup_stubs/gpu_abi.h"
#include "startup_stubs/gpu_memory.h"
#include "startup_stubs/gpu_compute_trace.h"
#include "startup_stubs/gpu_ambient.h"
#include "startup_stubs/gpu_memory_arena.h"
#include "startup_stubs/gpu_packed_light.h"
#include "startup_stubs/gpu_table_constants.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace x2w_vbind {

using x2w_gpu::GpuResourceType;
using x2w_gpu::GpuRoot;
using x2w_gpu::GpuRootIndexedView;
using x2w_gpu::GpuRootKind;
using x2w_gpu::GpuRootType;
using x2w_gpu::GpuUsedResource;
using x2w_gpu::GpuUserData;
namespace abi = x2w_gpu::abi;

inline constexpr std::uint32_t fresh_most = 1u << 18;     // Maximum slots in a single fresh table.
inline constexpr std::uint32_t mesh_groups_most = 65535;  // DispatchMesh: the host's limit per dimension

inline bool is_view(GpuRootKind k)
{
    return k == GpuRootKind::cbv || k == GpuRootKind::srv || k == GpuRootKind::uav;
}

struct VertexView { std::uint64_t address = 0; std::uint32_t size = 0, stride = 0; };   // D3D12_VERTEX_BUFFER_VIEW
struct IndexView { std::uint64_t address = 0; std::uint32_t size = 0, format = 0; };    // D3D12_INDEX_BUFFER_VIEW

// Separate graphics and compute root arguments, 32 input-assembler vertex slots and the title topology.
struct Roots {
    std::vector<GpuRoot> graphics, compute;
    VertexView vertex[32]{};
    IndexView index{};
    std::uint32_t topology = 0;  // console numbering
    bool topology_set = false;

    std::vector<GpuRoot>& of(bool is_compute) { return is_compute ? compute : graphics; }
    const std::vector<GpuRoot>& of(bool is_compute) const { return is_compute ? compute : graphics; }
    const GpuRoot* at(bool is_compute, std::uint32_t slot) const
    {
        const auto& r = of(is_compute);
        return slot < r.size() ? &r[slot] : nullptr;
    }
    GpuRoot& place(bool is_compute, std::uint32_t slot)
    {
        auto& r = of(is_compute);
        if (r.size() <= slot) r.resize(std::size_t{slot} + 1);
        return r[slot];
    }
    void set_table(bool is_compute, std::uint32_t slot, std::uint64_t handle)
    {
        GpuRoot& r = place(is_compute, slot);
        r.kind = GpuRootKind::table;
        r.handle = handle;
        r.address = 0;
        r.dwords.clear();
    }
    void set_view(bool is_compute, std::uint32_t slot, std::uint64_t address, GpuRootKind kind)
    {
        GpuRoot& r = place(is_compute, slot);
        r.kind = is_view(kind) ? kind : GpuRootKind::srv;
        r.address = address;
        r.handle = 0;
        r.dwords.clear();
    }
    // A parameter previously holding another kind starts empty and grows to offset + count.
    void set_constants(bool is_compute, std::uint32_t slot, const std::uint32_t* values, std::uint32_t count, std::uint32_t offset)
    {
        GpuRoot& r = place(is_compute, slot);
        if (r.kind != GpuRootKind::constants) {
            r.kind = GpuRootKind::constants;
            r.handle = r.address = 0;
            r.dwords.clear();
        }
        const std::size_t end = std::size_t{offset} + count;
        if (r.dwords.size() < end) r.dwords.resize(end, 0);
        for (std::uint32_t i = 0; i < count; ++i) r.dwords[std::size_t{offset} + i] = values[i];
    }
    void set_vertex(std::uint32_t start, std::uint32_t count, const VertexView* views)
    {
        for (std::uint32_t i = 0; i < count && std::size_t{start} + i < 32; ++i)
            vertex[start + i] = views ? views[i] : VertexView{};
    }
    void set_index(const IndexView* view) { index = view ? *view : IndexView{}; }
    void clear() { *this = Roots{}; }
};

// ---- topology -------------------------------------------------------------------
inline constexpr std::uint32_t topology_triangle_fan = 6;
inline constexpr std::uint32_t topology_rectangles = 17;
inline constexpr std::uint32_t topology_line_loop = 18;
inline constexpr std::uint32_t topology_quads = 19;
inline constexpr std::uint32_t topology_quad_strip = 20;
// The host topology for the title's: the console-only ones become triangle lists (4).
inline std::uint32_t host_topology(std::uint32_t title)
{
    switch (title) {
    case topology_rectangles:
    case topology_triangle_fan:
    case topology_line_loop:
    case topology_quads:
    case topology_quad_strip: return 4;
    default: return title;
    }
}

// ---- user data ---------------------------------------------------------------------
// A heap the title bound: its title GPU start, its descriptor count, and whether it holds samplers.
struct HeapSpan {
    std::uint64_t start = 0;
    std::uint32_t count = 0;
    bool samplers = false;
    std::uint32_t step() const { return samplers ? abi::sampler_increment : abi::view_increment; }
    std::optional<std::uint32_t> index(std::uint64_t handle) const
    {
        if (!start || handle < start) return std::nullopt;
        const std::uint64_t i = (handle - start) / step();
        return i < count ? std::optional<std::uint32_t>(static_cast<std::uint32_t>(i)) : std::nullopt;
    }
};
// The bound heap holding `handle`.
inline const HeapSpan* heap_of(const HeapSpan* heaps, std::size_t n, std::uint64_t handle)
{
    for (std::size_t i = 0; i < n; ++i)
        if (heaps[i].start && heaps[i].index(handle)) return &heaps[i];
    return nullptr;
}

// Why a user-data dword came out 0 (counted by the caller).
enum class UserMiss : std::uint8_t { none, table_outside_heaps, root_not_set, unknown_host_word, view_address_missing };

// The value of a user-data dword that names a title root argument: a table's byte offset in its heap (console
// units), a constant's value. Other(...) codes are the host's (host_word), 0 here.
inline std::uint32_t user_value(const Roots& roots, bool is_compute, std::uint32_t slot, GpuRootType root, std::uint32_t dword,
                                const HeapSpan* heaps, std::size_t heap_count, UserMiss* miss = nullptr)
{
    if (root == GpuRootType::other) return 0;
    const GpuRoot* r = roots.at(is_compute, slot);
    if (r && r->kind == GpuRootKind::table && root == GpuRootType::table) {
        if (const HeapSpan* h = heap_of(heaps, heap_count, r->handle)) return static_cast<std::uint32_t>(r->handle - h->start);
        if (miss) *miss = UserMiss::table_outside_heaps;
        return 0;
    }
    if (r && r->kind == GpuRootKind::constants && root == GpuRootType::constants)
        return dword < r->dwords.size() ? r->dwords[dword] : 0;
    if (miss) *miss = UserMiss::root_not_set;
    return 0;
}

// The bias the bound raw view of root parameter `slot` begins before the title's address (0-15), as the caller
// bound it (root_srv_table / root_uav): taken from what was bound, not found again.
struct BoundBias {
    std::uint32_t slot;
    std::uint32_t bias;
};
inline std::uint32_t bias_of(const std::vector<BoundBias>& bound, std::uint32_t slot)
{
    for (const auto& b : bound)
        if (b.slot == slot) return b.bias;
    return 0;
}

// A host-computed user-data dword. `view_address(slot, dword)` gives the title address of the buffer view at dword
// `dword` of the table in root parameter `slot` (0: none).
template<class ViewAddress>
std::uint32_t host_word(const Roots& roots, bool is_compute, std::uint32_t code, std::uint32_t slot, std::uint32_t dword,
                        const std::vector<BoundBias>& biases, std::uint32_t memory_base, ViewAddress&& view_address,
                        UserMiss* miss = nullptr)
{
    if (code == abi::user_memory_base) return memory_base;
    if (code == abi::user_dispatch_threads_x) return 0;  // all of every group, unless the dispatch says otherwise
    const GpuRoot* r = roots.at(is_compute, slot);
    if (code == abi::user_root_address || code == abi::user_root_address + 1) {
        if (!r || !is_view(r->kind)) return 0;
        return code == abi::user_root_address ? static_cast<std::uint32_t>(r->address) : static_cast<std::uint32_t>(r->address >> 32);
    }
    if (code == abi::user_view_bias) return r && is_view(r->kind) ? bias_of(biases, slot) : 0;
    if (code != abi::user_view_address && code != abi::user_view_address + 1) {
        if (miss) *miss = UserMiss::unknown_host_word;
        return 0;
    }
    const std::uint64_t address = r && r->kind == GpuRootKind::table ? static_cast<std::uint64_t>(view_address(slot, dword)) : 0;
    if (!address && miss) *miss = UserMiss::view_address_missing;
    return code == abi::user_view_address ? static_cast<std::uint32_t>(address) : static_cast<std::uint32_t>(address >> 32);
}

// The fresh offset a table got for this draw (tables()), by root parameter. UINT32_MAX: no room (drop the draw).
struct MadeTable {
    std::uint32_t slot;
    std::uint32_t offset;
};

// The stage's user-data dwords: `dwords` of them, every word the shader reads filled in. Words whose index lies
// past `dwords` are left out (the pipeline's root has only `dwords`).
template<class ViewAddress>
std::vector<std::uint32_t> fill_user_data(const Roots& roots, bool is_compute, const GpuUserData* words, std::size_t word_count,
                                          std::uint32_t dwords, const std::vector<MadeTable>& made, const HeapSpan* heaps,
                                          std::size_t heap_count, const std::vector<BoundBias>& biases,
                                          std::uint32_t memory_base, ViewAddress&& view_address, unsigned* misses = nullptr)
{
    std::vector<std::uint32_t> values(dwords, 0);
    for (std::size_t i = 0; i < word_count; ++i) {
        const GpuUserData& u = words[i];
        if (u.index >= dwords) continue;
        UserMiss miss = UserMiss::none;
        if (u.root == GpuRootType::other) {
            values[u.index] = host_word(roots, is_compute, u.other, u.slot, u.dword, biases, memory_base, view_address, &miss);
        } else {
            const auto fresh = std::find_if(made.begin(), made.end(), [&](const MadeTable& m) {
                return m.slot == u.slot && u.root == GpuRootType::table;
            });
            values[u.index] = fresh != made.end() ? fresh->offset
                                                  : user_value(roots, is_compute, u.slot, u.root, u.dword, heaps, heap_count, &miss);
        }
        if (miss != UserMiss::none && misses) ++*misses;
    }
    return values;
}

// ---- tables made afresh --------------------------------------------------------
// A table the draw gets in dynamic slots: root parameter, title GPU handle, slot count, and the entries root state
// selects (root_indexed_views), by descriptor index in the table, sorted.
struct FreshPlan {
    std::uint32_t slot;
    std::uint64_t table;
    std::uint32_t n;
    std::vector<std::uint32_t> selected;
};
enum class PlanSkip : std::uint8_t { none, sampler_or_unaligned, too_long };
struct TablePlan {
    std::vector<FreshPlan> fresh;              // tables made afresh (constant buffers; past the host heap)
    std::vector<std::size_t> mirrored;         // indices into `used` whose entry is mirrored in place (table_entry)
    std::vector<std::pair<std::uint32_t, PlanSkip>> skipped;  // Fresh tables left as the title's
};
inline bool in_table(const GpuUsedResource& u) { return u.root == GpuRootType::table; }

// `host_count(heap)`: how many title slots the host heap holds (normally heap.count). Tables holding a CBV are made
// afresh (the host wants 256-byte aligned CBVs, the console 16), and tables reaching past the host's slots.
template<class HostCount>
TablePlan plan_tables(const Roots& roots, bool is_compute, const GpuUsedResource* used, std::size_t used_count,
                      const GpuRootIndexedView* indexed, std::size_t indexed_count, const HeapSpan* heaps, std::size_t heap_count,
                      HostCount&& host_count)
{
    TablePlan plan;
    std::vector<std::uint32_t> fresh;
    const auto has = [&](std::uint32_t s) { return std::find(fresh.begin(), fresh.end(), s) != fresh.end(); };
    for (std::size_t i = 0; i < used_count; ++i)
        if (in_table(used[i]) && used[i].kind == GpuResourceType::cbv && !has(used[i].slot)) fresh.push_back(used[i].slot);
    for (std::size_t i = 0; i < used_count; ++i) {
        const GpuUsedResource& u = used[i];
        if (!in_table(u) || has(u.slot)) continue;
        const GpuRoot* r = roots.at(is_compute, u.slot);
        if (!r || r->kind != GpuRootKind::table) continue;
        const std::uint64_t end = r->handle + std::uint64_t{u.table_offset_dwords} * 4;
        const HeapSpan* h = heap_of(heaps, heap_count, r->handle);
        if (!h) continue;
        const auto index = h->index(end);
        if (!index) continue;
        if (*index >= host_count(*h)) fresh.push_back(u.slot);
    }
    std::sort(fresh.begin(), fresh.end());
    for (std::size_t i = 0; i < used_count; ++i)
        if (in_table(used[i]) && !has(used[i].slot)) plan.mirrored.push_back(i);
    for (const std::uint32_t slot : fresh) {
        const GpuRoot* r = roots.at(is_compute, slot);
        if (!r || r->kind != GpuRootKind::table) continue;
        const std::uint64_t table = r->handle;
        const HeapSpan* h = heap_of(heaps, heap_count, table);
        if (!h) continue;
        if (h->samplers || (table - h->start) % abi::view_increment != 0) {
            plan.skipped.emplace_back(slot, PlanSkip::sampler_or_unaligned);
            continue;
        }
        std::vector<std::uint32_t> selected;
        for (std::size_t k = 0; k < indexed_count; ++k) {
            const GpuRootIndexedView& x = indexed[k];
            if (x.table_slot != slot) continue;
            const std::uint32_t constant = user_value(roots, is_compute, x.constant_slot, GpuRootType::constants,
                                                      x.constant_dword, heaps, heap_count);
            // The shader adds i32 byte offsets and shifts by 5 (u32 wrapping).
            selected.push_back((x.byte_offset + (constant << x.shift)) / abi::view_increment);
        }
        std::sort(selected.begin(), selected.end());
        selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
        std::uint32_t most = selected.empty() ? 0 : selected.back();
        for (std::size_t i = 0; i < used_count; ++i)
            if (in_table(used[i]) && used[i].slot == slot) most = (std::max)(most, used[i].table_offset_dwords / 8);
        const std::uint32_t n = most + 1;
        const std::uint64_t last = table + std::uint64_t{n - 1} * abi::view_increment;
        if (n > fresh_most || last < table || !h->index(last)) {
            plan.skipped.emplace_back(slot, PlanSkip::too_long);
            continue;
        }
        plan.fresh.push_back({slot, table, n, std::move(selected)});
    }
    return plan;
}

// ---- RecordHash: a hash of descriptor records' first three words -------------------------------------
inline std::uint64_t fold64(std::uint64_t x, std::uint64_t y)
{
#if defined(_MSC_VER) && defined(_M_X64)
    std::uint64_t high = 0;
    const std::uint64_t low = _umul128(x, y, &high);
    return low ^ high;
#else
    const unsigned __int128 p = static_cast<unsigned __int128>(x) * y;
    return static_cast<std::uint64_t>(p) ^ static_cast<std::uint64_t>(p >> 64);
#endif
}
struct RecordHash {
    std::uint64_t value = 0;
    void add_record(const std::uint32_t (&record)[8])
    {
        add(record[0], record[1], record[2]);
        add(record[3], record[4], record[5]);
        add(record[6], record[7], 0);
    }
    void add(std::uint32_t a, std::uint32_t b, std::uint32_t c)
    {
        const std::uint64_t words = std::uint64_t{a} << 32 | b;
        const std::uint64_t first = fold64(value ^ words ^ 0x2d358dccaa6c78a5ull, 0x8bb84b93962eacc9ull ^ std::uint64_t{c});
        const std::uint64_t rotated = value << 23 | value >> 41;
        value = fold64(first ^ 0x4b33a62ed433d4a3ull, 0x4d5a2da51de1aa47ull ^ rotated) ^ first;
    }
};

// ---- dispatch arithmetic --------------------------------------------------------------------------------------------
// The groups of `size` threads that hold `threads`.
inline void dispatch_groups(const std::uint32_t threads[3], const std::uint32_t size[3], std::uint32_t out[3])
{
    for (int i = 0; i < 3; ++i) out[i] = size[i] ? (threads[i] + size[i] - 1) / size[i] : 0;
}
// DispatchMesh(x, 1, 1): the console's mesh programs count groups in x only; the host takes 65,535 a dimension.
inline std::uint32_t mesh_groups(std::uint32_t x) { return (std::min)(x, mesh_groups_most); }

// ---- raw view bounding ------------------------------------------------------------
// A raw view for a root view at byte `offset` of its host buffer with `left` bytes after it: from the 16 bytes
// below when the shader adds the bias (biased), else only when on 16 bytes. False: a null view.
struct RawWindow {
    std::uint64_t first = 0;   // byte offset of the view in the buffer (a multiple of 16)
    std::uint64_t bytes = 0;   // bytes the view covers
    std::uint32_t bias = 0;    // what the shader adds (USER_VIEW_BIAS)
};
inline bool raw_window(std::uint64_t offset, std::uint64_t left, bool biased, RawWindow* out)
{
    if (!left) return false;
    const std::uint64_t bias = offset & 15;
    if (!biased && bias) return false;  // "a root SRV not on 16 bytes reads zeros"
    out->first = offset - bias;
    out->bytes = left + bias;
    out->bias = static_cast<std::uint32_t>(bias);
    return true;
}
// A raw R32_TYPELESS view spans whole dwords with a minimum of one element.
inline std::uint32_t raw_elements(std::uint64_t bytes)
{
    const std::uint64_t n = bytes / 4;
    return n < 1 ? 1u : n > 0xffffffffull ? 0xffffffffu : static_cast<std::uint32_t>(n);
}
// A root CBV covers up to 64 KiB, bounded by the end of its buffer.
inline std::uint64_t root_cbv_bytes(std::uint64_t left) { return (std::min)(left, std::uint64_t{1} << 16); }
// The size of a CBV for `bytes` of constants: 256-byte steps, at most 64 KiB (D3D12's limits).
inline std::uint32_t constant_view_bytes(std::uint64_t bytes)
{
    const std::uint64_t b = (std::min)(bytes, std::uint64_t{1} << 16);
    return static_cast<std::uint32_t>((b + 255) & ~std::uint64_t{255});
}

// Bound a title buffer access by the allocation end, contiguous desktop window and caller limit. Desktop windows
// can span 512 MiB, so using the window end alone can let a raw UAV overwrite neighboring title allocations.
// Without a title buffer record only unknown_most bytes may be exposed; zero refuses the access. The extent
// callback locates the title allocation and the window callback resolves the bounded range to a host buffer and
// offset.
struct TitleBufferExtent {
    std::uint64_t base{}, end{};
    std::uintptr_t identity{};

    bool consider(std::uint64_t address, std::uint64_t length, std::uint64_t start,
                  std::uint64_t bytes, std::uintptr_t id)
    {
        if (!start || address < start || address - start >= bytes || bytes > UINT64_MAX - start) return false;
        const auto limit = start + bytes;
        const bool covers = length <= limit - address;
        const bool previous_covers = base && length <= end - address;
        if (base && (covers != previous_covers ? !covers :
                     start < base || (start == base && id <= identity))) return false;
        base = start;
        end = limit;
        identity = id;
        return true;
    }
};

struct Found {
    void* buffer = nullptr;        // the desktop window (ID3D12Resource*)
    std::uint64_t offset = 0;      // byte offset of `address` in it
    std::uint64_t left = 0;        // bytes usable from `address` on
    bool known_extent = false;     // a title buffer record bounds `left`
    bool cut_by_window = false;    // the title buffer runs on past where the window allows
};

template<class Extent, class Window>
bool find_bounded(std::uint64_t address, std::uint64_t len, std::uint64_t most, std::uint64_t unknown_most, Extent&& extent,
                  Window&& window, Found* out)
{
    *out = Found{};
    if (!address || !len) return false;
    std::uint64_t base = 0, end = 0;
    std::uint64_t left = 0;
    if (extent(address, &base, &end) && end > address) {
        left = end - address;
        out->known_extent = true;
    } else {
        left = unknown_most;  // no record: only the caller's allowance for unknown memory
    }
    if (left > most) left = most;
    if (left > ~0ull - address) left = ~0ull - address;  // an unbounded allowance must not wrap the address space
    if (left < len) return false;
    void* buffer = nullptr;
    std::uint64_t offset = 0;
    if (!window(address, len, &buffer, &offset) || !buffer) return false;
    // The whole bounded range in one window, else the longest prefix of it that is. Windows break only on page
    // boundaries (64 KiB chunk pages, 256 MiB window steps, allocation ends), so the prefix ends on the last 4 KiB
    // boundary that still answers: a binary search over boundaries (at most ~17 tries for 512 MiB; desktop_range
    // caches its answers per frame).
    void* b = nullptr;
    std::uint64_t o = 0;
    if (left != len && (!window(address, left, &b, &o) || b != buffer)) {
        constexpr std::uint64_t page = 4096;
        std::uint64_t good_end = address + len;
        const std::uint64_t bad_end = address + left;
        std::uint64_t lo = (good_end + page - 1) / page, hi = (bad_end - 1) / page;
        while (lo <= hi) {
            const std::uint64_t mid = lo + (hi - lo) / 2;
            if (mid * page > good_end && window(address, mid * page - address, &b, &o) && b == buffer) {
                good_end = mid * page;
                lo = mid + 1;
            } else if (mid * page <= good_end) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        left = good_end - address;
        out->cut_by_window = true;
    }
    out->buffer = buffer;
    out->offset = offset;
    out->left = left;
    return true;
}


// ---- per-list host-side bookkeeping ---------
struct Dynamic {
    x2w_gpu::GpuHostHeap* heap;
    std::uint32_t first, n;
};
struct RootSlot {
    std::uint32_t index;          // the dynamic slot holding the one-view table
    ID3D12Resource* window;       // the desktop window it shows (null: a null view)
    std::uint32_t bias;           // what the shader adds (biased SRVs, UAVs)
};
struct KeyHash {
    std::size_t operator()(const std::pair<const void*, std::uint64_t>& k) const
    {
        return std::hash<std::uint64_t>()(reinterpret_cast<std::uint64_t>(k.first) * 1099511628211ull ^ k.second);
    }
};
struct FreshKey {
    std::uint64_t table;
    std::uint32_t n;
    std::uint64_t hash;
    bool operator==(const FreshKey& o) const { return table == o.table && n == o.n && hash == o.hash; }
};
struct FreshHash {
    std::size_t operator()(const FreshKey& k) const
    {
        return std::hash<std::uint64_t>()(k.table ^ (std::uint64_t{k.n} << 40) ^ k.hash * 31);
    }
};

}  // namespace x2w_vbind

// One command list's translated state (declared in gpu_abi.h), hung off the list wrapper's own().gpu[0]
// (gpu_hooks.h). Made at the first hook call on the list, freed by gpu_list_released. A list records on one
// thread at a time.
namespace x2w_gpu {
struct GpuListState {
    x2w_vbind::Roots roots;
    ambient::Role light_role = ambient::Role::none;
    ambient::Views light_views{};
    UINT64 light_depth_width = 0;
    UINT light_depth_height = 0;
    LONG trace_request = 0;
    const GpuShader* trace_shader = nullptr;
    UINT material_raw_slot = 0, material_base = 0;
    const GpuShader* material_shader = nullptr;
    GpuHostHeap* material_heap = nullptr;
    x2w_compute_trace::Bindings trace_bindings;
    std::vector<x2w_compute_trace::Binding> pixel_bindings;
    const GpuShader* pixel_capture_shader = nullptr;
    const GpuPipeline* pixel_capture_pipeline = nullptr;
    std::shared_ptr<x2w_compute_trace::Capture> trace_current;
    std::vector<std::shared_ptr<x2w_compute_trace::Capture>> trace_captures;
    bool vertex_hooked = false;      // slots 47/48 reached us (else X2W's Pass.vb / Pass.ib are read)
    void* pipeline = nullptr;        // slot 25: the title pipeline wrapper
    void* heaps[2]{};                // slot 32: view heap, sampler heap (title wrappers)
    // What the host list has (only trusted with X2W_GPU_BIND_CACHE=1, or until gpu_bind_forget_compute).
    ID3D12DescriptorHeap* host_heaps[2]{};
    ID3D12RootSignature* host_roots[2]{};  // graphics, compute
    ID3D12PipelineState* host_pso = nullptr;
    // Made in this recording, given back with `dynamic`.
    std::unordered_map<x2w_vbind::FreshKey, std::uint32_t, x2w_vbind::FreshHash> fresh_tables;
    TableConstants table_constants;
    std::unordered_map<std::pair<const void*, std::uint64_t>, x2w_vbind::RootSlot, x2w_vbind::KeyHash> root_srv_slots, root_uav_slots;
    std::vector<x2w_vbind::Dynamic> dynamic;
    std::vector<std::shared_ptr<x2w_depth_twins::Twins>> depth_refs;
    std::vector<std::shared_ptr<x2w_vmemory::Recording>> memories;
    std::vector<x2w_vmemory::CopyBack> memory_back;
    std::vector<std::shared_ptr<x2w_packed_light::Copy>> light_copies;
    std::vector<x2w_depth_twins::Twins*> writable_depth;
    // The fences of this recording's submissions: its slots go back once all have passed (X2W does not wait).
    std::vector<std::pair<ID3D12Fence*, std::uint64_t>> submitted;
};
}  // namespace x2w_gpu
