
bool heap_raw_on()
{
    static const bool on = [] {
        char v[8] = "";
        const DWORD n = GetEnvironmentVariableA("X2W_HEAP_RAW", v, sizeof v);
        const bool off = n && n < sizeof v && v[0] == '0';
        log("x2w-heapraw X2W_HEAP_RAW=%s: raw V#s in the title heap %s for `heapraw 1` compute translations", off ? "0" : "1 (default)",
            off ? "bind the empty view (off)" : "resolve through the table of the runtime's own V#s");
        return !off;
    }();
    return on;
}

struct HeapRawPending { u64 base; std::uint32_t bytes; };
SRWLOCK heap_raw_pending_lock = SRWLOCK_INIT;
std::vector<HeapRawPending> heap_raw_pending;
LONG64 heap_raw_noted, heap_raw_dropped;
constexpr std::size_t heap_raw_pending_max = 1u << 20;
volatile LONG heap_raw_rescan;

// The table size: heap_raw_map_entries, or (tests) X2W_HEAP_RAW_TEST_ENTRIES, a power of two from 4 to 2^20.
std::uint32_t heap_raw_entries()
{
    static const std::uint32_t n = [] {
        char v[16] = "";
        const DWORD got = GetEnvironmentVariableA("X2W_HEAP_RAW_TEST_ENTRIES", v, sizeof v);
        const unsigned long t = got && got < sizeof v ? std::strtoul(v, nullptr, 10) : 0;
        if (t >= 4 && t <= (1u << 20) && !(t & (t - 1))) {
            log("x2w-heapraw X2W_HEAP_RAW_TEST_ENTRIES=%lu: a test-size table", t);
            return static_cast<std::uint32_t>(t);
        }
        return x2w_startup::heap_raw_map_entries;
    }();
    return n;
}

// write_buffer_descriptor (d3d12_memory_srv.inl), for the V# words it wrote.
void heap_raw_note(const std::uint32_t* words)
{
    if (!heap_raw_on()) return;
    x2w_startup::RawBufferWords v;
    if (!x2w_startup::raw_buffer_words(words, v)) return;
    AcquireSRWLockExclusive(&heap_raw_pending_lock);
    const bool room = heap_raw_pending.size() < heap_raw_pending_max;
    if (room) heap_raw_pending.push_back({v.base, v.bytes});
    else heap_raw_rescan = 1;  // the next drain reads the whole heap (under this lock: the drain clears it with the swap)
    ReleaseSRWLockExclusive(&heap_raw_pending_lock);
    InterlockedIncrement64(room ? &heap_raw_noted : &heap_raw_dropped);
}

ID3D12Resource* heap_raw_map;
std::uint32_t* heap_raw_words;  // the GPU copy, heap_raw_entries() x 4 (write only)
struct HeapRawSlot {
    std::uint32_t lo = 0, hi_window = 0, bytes = 0, offset = 0;  // the GPU entry's words
    std::uint32_t seen = 0;                                        // sweep pass of the last sighting
    LONG tried = 0;                                                // frame of the last resolution attempt
    std::uint32_t batch = 0;                                       // heap_raw_batch_id when queued for the GPU copy
    std::uint32_t checked = 0;                                     // sweep pass of the last resolution or revalidation
    UINT window() const { return hi_window >> 16; }
    u64 base() const { return u64{lo} | (u64{hi_window & 0xffff} << 32); }
};
std::vector<HeapRawSlot> heap_raw_shadow;
constexpr UINT heap_raw_tombstone = 0xffff, heap_raw_refused = 0xfffe;  // both a miss for the program (window < 256)
std::uint32_t heap_raw_live, heap_raw_refused_live;  // occupied entries (resolved + refused), and the refused among them
std::uint32_t heap_raw_pass = 2;                     // the sweep pass in progress
// The last complete sweep pass (heap_raw_bits_pass): one bit per table entry the pass found in the heap. An entry's
// sighting is max(seen, that pass) when its bit is set (heap_raw_seen).
std::vector<std::uint64_t> heap_raw_bits;
std::uint32_t heap_raw_bits_pass = 0;
std::uint32_t heap_raw_seen(std::uint32_t slot)
{
    const std::uint32_t s = heap_raw_shadow[slot].seen;
    const bool bit = !heap_raw_bits.empty() && (heap_raw_bits[slot >> 6] >> (slot & 63)) & 1;
    return bit && heap_raw_bits_pass > s ? heap_raw_bits_pass : s;
}
std::vector<HeapRawPending> heap_raw_backlog;  // heap-sweep finds past the per-drain budget
constexpr std::size_t heap_raw_backlog_max = 1u << 19;
std::vector<ID3D12Resource*> heap_raw_windows;  // index = window view - heap_raw_window_first
LONG64 heap_raw_paged, heap_raw_paged_full, heap_raw_unaligned_resolved;  // paged and unaligned ranges resolved; paged refused (no records left)
LONG64 heap_raw_resolved, heap_raw_unaligned, heap_raw_unmapped, heap_raw_past_view, heap_raw_no_window, heap_raw_displaced,
    heap_raw_displaced_recent, heap_raw_evicted, heap_raw_scans, heap_raw_deferred, heap_raw_check_mismatch, heap_raw_passes;
// Heartbeat counters (report_heap_raw, every 5 s), reset at each report.
LONG64 heap_raw_hb_drains, heap_raw_hb_ticks, heap_raw_hb_max_ticks, heap_raw_hb_resolved, heap_raw_hb_evicted, heap_raw_hb_sweep_ticks,
    heap_raw_hb_sweep_max_ticks, heap_raw_hb_displaced;
double heap_raw_ms(LONG64 ticks)
{
    static const double per_ms = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return static_cast<double>(f.QuadPart) / 1000.0;
    }();
    return static_cast<double>(ticks) / per_ms;
}
bool heap_raw_env(const char* name)
{
    char v[4] = "";
    const DWORD n = GetEnvironmentVariableA(name, v, sizeof v);
    return n && n < sizeof v && v[0] == '1';
}
// X2W_HEAP_RAW_TIMES=1 (benchmarks, tests/startup_cx_async.cpp rawheap-bench): one log line per drain with work.
bool heap_raw_times()
{
    static const bool on = heap_raw_env("X2W_HEAP_RAW_TIMES");
    return on;
}
// X2W_HEAP_RAW_CHECK=1 (tests, watched runs): every page-cache resolution of a new range is compared with
// desktop_range(base, bytes) (not the revalidations: their page budget would not bound those calls).
bool heap_raw_revalidating;
bool heap_raw_check()
{
    static const bool on = heap_raw_env("X2W_HEAP_RAW_CHECK");
    return on;
}

// The index of the window's raw SRV, made on first sight. UINT_MAX: no room (heap_raw_windows_max).
std::vector<u64> heap_raw_window_views;  // each window's view bytes
UINT heap_raw_window(ID3D12Resource* window, u64* view_bytes)
{
    for (UINT i = 0; i < heap_raw_windows.size(); ++i)
        if (heap_raw_windows[i] == window) {
            *view_bytes = heap_raw_window_views[i];
            return i;
        }
    const u64 width = window->GetDesc().Width;
    *view_bytes = (width < x2w_startup::heap_raw_window_view_bytes ? width : x2w_startup::heap_raw_window_view_bytes) & ~3ull;
    if (heap_raw_windows.size() >= x2w_startup::heap_raw_windows_max) return UINT_MAX;
    const UINT i = static_cast<UINT>(heap_raw_windows.size());
    bounded_raw_srv(window, 0, *view_bytes,
                    D3D12_CPU_DESCRIPTOR_HANDLE{draw_heap->GetCPUDescriptorHandleForHeapStart().ptr +
                                                SIZE_T{x2w_startup::heap_raw_window_first + i} * draw_heap_increment});
    heap_raw_windows.push_back(window);
    heap_raw_window_views.push_back(*view_bytes);
    log("x2w-heapraw window %u: desktop va 0x%llx, %llu bytes (view %llu)", i, window->GetGPUVirtualAddress(), width, *view_bytes);
    return i;
}

// Writes entry `slot` in the shadow now and in the GPU copy at the next heap_raw_flush (an upload heap:
// write-combined memory, so the order the GPU may see is fenced with _mm_sfence, three fences per flush instead of
// two per entry). A slot that was free on the GPU (z == 0): x, y, w, fence, then z. A slot in use or a tombstone
// (z != 0, which keeps the probe chains through it intact): the window is set to heap_raw_tombstone first (a value
// the program never accepts), fence, x, z, w, fence, then the window. A slot written twice in one batch flushes the
// batch first.
struct HeapRawGpuWrite { std::uint32_t slot; bool was_free; };
std::vector<HeapRawGpuWrite> heap_raw_batch;
std::uint32_t heap_raw_batch_id = 1;
void heap_raw_flush()
{
    if (heap_raw_batch.empty()) return;
    volatile std::uint32_t* const words = heap_raw_words;
    for (const HeapRawGpuWrite& w : heap_raw_batch)
        if (!w.was_free) words[u64{w.slot} * 4 + 1] = (heap_raw_shadow[w.slot].hi_window & 0xffff) | (heap_raw_tombstone << 16);
    _mm_sfence();
    for (const HeapRawGpuWrite& w : heap_raw_batch) {
        const HeapRawSlot& e = heap_raw_shadow[w.slot];
        volatile std::uint32_t* g = words + u64{w.slot} * 4;
        g[0] = e.lo;
        g[3] = e.offset;
        if (w.was_free) g[1] = e.hi_window;
        else g[2] = e.bytes;
    }
    _mm_sfence();
    for (const HeapRawGpuWrite& w : heap_raw_batch) {
        const HeapRawSlot& e = heap_raw_shadow[w.slot];
        if (w.was_free) words[u64{w.slot} * 4 + 2] = e.bytes;
        else words[u64{w.slot} * 4 + 1] = e.hi_window;
    }
    _mm_sfence();
    heap_raw_batch.clear();
    ++heap_raw_batch_id;
}
void heap_raw_write(std::uint32_t slot, u64 base, std::uint32_t bytes, UINT window, std::uint32_t offset)
{
    HeapRawSlot& s = heap_raw_shadow[slot];
    if (s.batch == heap_raw_batch_id) heap_raw_flush();
    s.batch = heap_raw_batch_id;
    heap_raw_batch.push_back({slot, s.bytes == 0});
    s.lo = static_cast<std::uint32_t>(base);
    s.hi_window = static_cast<std::uint32_t>((base >> 32) & 0xffff) | (window << 16);
    s.bytes = bytes;
    s.offset = offset;
}
// Only the window changes (a tombstone): one store, fenced by the caller.
void heap_raw_set_window(std::uint32_t slot, UINT window)
{
    HeapRawSlot& s = heap_raw_shadow[slot];
    if (s.batch == heap_raw_batch_id) heap_raw_flush();
    s.hi_window = (s.hi_window & 0xffff) | (window << 16);
    heap_raw_words[u64{slot} * 4 + 1] = s.hi_window;
}

bool heap_raw_ensure()
{
    if (heap_raw_map) return true;
    static bool failed;
    if (failed) return false;
    D3D12_HEAP_PROPERTIES up{D3D12_HEAP_TYPE_UPLOAD};
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = u64{heap_raw_entries()} * 16;
    rd.Height = rd.DepthOrArraySize = rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(device_of(the_device)->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                              IID_PPV_ARGS(&heap_raw_map))) ||
        FAILED(heap_raw_map->Map(0, nullptr, reinterpret_cast<void**>(&heap_raw_words)))) {
        if (heap_raw_map) heap_raw_map->Release();
        heap_raw_map = nullptr;
        failed = true;
        log("x2w-heapraw table not created: raw V#s bind the empty view");
        return false;
    }
    std::memset(heap_raw_words, 0, static_cast<std::size_t>(rd.Width));
    heap_raw_shadow.assign(heap_raw_entries(), HeapRawSlot{});
    log("x2w-heapraw table: %u entries (%llu bytes), %u probes, sweep of the heap in %u frames, %u heap finds resolved per drain; windows "
        "from draw heap view %u",
        heap_raw_entries(), rd.Width, x2w_startup::heap_raw_probes, x2w_startup::heap_raw_sweep_frames, x2w_startup::heap_raw_resolve_budget,
        x2w_startup::heap_raw_window_first);
    return true;
}

void heap_raw_counters()
{
    log("x2w-heapraw ranges: noted %lld (past a full pending list %lld, read again by a rescan); resolved %lld "
        "(base not 4-aligned %lld, paged %lld); not resolved: base not 4-aligned %lld, not in windows %lld, past the window view %lld, no "
        "window view left %lld, page records full %lld; displaced %lld (recent %lld), evicted %lld; %u entries (%u refused), "
        "%zu windows, %zu waiting (deferred %lld), sweep passes %lld, heap reads %lld, check mismatches %lld",
        heap_raw_noted, heap_raw_dropped, heap_raw_resolved, heap_raw_unaligned_resolved, heap_raw_paged, heap_raw_unaligned,
        heap_raw_unmapped, heap_raw_past_view, heap_raw_no_window, heap_raw_paged_full,
        heap_raw_displaced, heap_raw_displaced_recent, heap_raw_evicted, heap_raw_live, heap_raw_refused_live, heap_raw_windows.size(),
        heap_raw_backlog.size(), heap_raw_deferred, heap_raw_passes, heap_raw_scans, heap_raw_check_mismatch);
}

// The table entry of (base, bytes), or UINT_MAX. `place` gets where an insert goes: the first free or tombstone slot
// on the chain, else the occupied slot seen least recently (UINT_MAX only for an empty probe range, never).
std::uint32_t heap_raw_find(u64 base, std::uint32_t bytes, std::uint32_t* place)
{
    const std::uint32_t mask = heap_raw_entries() - 1, lo = static_cast<std::uint32_t>(base), hi = static_cast<std::uint32_t>((base >> 32) & 0xffff);
    const std::uint32_t h = x2w_startup::raw_map_hash(lo, hi, bytes);
    std::uint32_t free = UINT_MAX, oldest = UINT_MAX;
    for (std::uint32_t i = 0; i < x2w_startup::heap_raw_probes; ++i) {
        const std::uint32_t at = (h + i) & mask;
        const HeapRawSlot& s = heap_raw_shadow[at];
        if (!s.bytes) {  // the end of the chain (the program stops here too)
            if (free == UINT_MAX) free = at;
            break;
        }
        if (s.window() == heap_raw_tombstone) {
            if (free == UINT_MAX) free = at;
            continue;
        }
        if (s.lo == lo && (s.hi_window & 0xffff) == hi && s.bytes == bytes) return at;
        if (oldest == UINT_MAX || heap_raw_seen(at) < heap_raw_seen(oldest)) oldest = at;
    }
    if (place) *place = free != UINT_MAX ? free : oldest;
    return UINT_MAX;
}

// Per-frame 64 KiB page cache for resolution: the window and the window offset of each page's start.
struct HeapRawPage {
    u64 page = 0;
    LONG frame = -1;
    ID3D12Resource* window = nullptr;  // nullptr: the page is not in one window
    u64 offset = 0;
};
std::vector<HeapRawPage> heap_raw_pages;
LONG64 heap_raw_page_lookups;  // desktop_range calls the page cache made
const HeapRawPage& heap_raw_page(u64 page)
{
    if (heap_raw_pages.empty()) heap_raw_pages.resize(1u << 15);
    const std::size_t mask = heap_raw_pages.size() - 1;
    std::size_t at = static_cast<std::size_t>((page * 0x9e3779b97f4a7c15ull) >> 40) & mask;
    for (std::size_t i = 0; i < 8; ++i, at = (at + 1) & mask) {  // linear probes; a stale or full run reuses the first
        HeapRawPage& p = heap_raw_pages[at];
        if (p.frame == frame_number && p.page == page) return p;
        if (p.frame != frame_number) break;
    }
    HeapRawPage& p = heap_raw_pages[at];
    p.page = page;
    p.frame = frame_number;
    p.window = nullptr;
    p.offset = 0;
    ++heap_raw_page_lookups;
    if (!desktop_range(page << 16, 4, &p.window, &p.offset)) p.window = nullptr;
    return p;
}

ID3D12Resource* heap_raw_pages_map;
volatile std::uint32_t* heap_raw_page_words;
std::vector<std::uint32_t> heap_raw_page_shadow;
std::uint32_t heap_raw_page_top;
std::multimap<std::uint32_t, std::uint32_t> heap_raw_page_free;  // count -> first
struct HeapRawFreed { LONG frame; std::uint32_t first, count; };
std::vector<HeapRawFreed> heap_raw_page_freed;
constexpr LONG heap_raw_page_quarantine = 8;
std::uint32_t heap_raw_page_count(u64 base, std::uint32_t bytes) { return static_cast<std::uint32_t>(((base & 0xffff) + bytes - 1) >> 16) + 1; }
std::uint32_t heap_raw_page_alloc(std::uint32_t count)
{
    if (!heap_raw_pages_map) {
        static bool failed;
        if (failed) return UINT_MAX;
        LARGE_INTEGER c0, c1;
        QueryPerformanceCounter(&c0);
        D3D12_HEAP_PROPERTIES up{D3D12_HEAP_TYPE_UPLOAD};
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = u64{x2w_startup::heap_raw_page_records} * 8;
        rd.Height = rd.DepthOrArraySize = rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        void* words = nullptr;
        if (FAILED(device_of(the_device)->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                                  IID_PPV_ARGS(&heap_raw_pages_map))) ||
            FAILED(heap_raw_pages_map->Map(0, nullptr, &words))) {
            if (heap_raw_pages_map) heap_raw_pages_map->Release();
            heap_raw_pages_map = nullptr;
            failed = true;
            log("x2w-heapraw page records not created: ranges over several windows stay refused");
            return UINT_MAX;
        }
        heap_raw_page_words = static_cast<std::uint32_t*>(words);
        heap_raw_page_shadow.assign(std::size_t{x2w_startup::heap_raw_page_records} * 2, 0);
        bounded_raw_srv(heap_raw_pages_map, 0, rd.Width,
                        D3D12_CPU_DESCRIPTOR_HANDLE{draw_heap->GetCPUDescriptorHandleForHeapStart().ptr +
                                                    SIZE_T{x2w_startup::heap_raw_pages_view} * draw_heap_increment});
        QueryPerformanceCounter(&c1);
        log("x2w-heapraw page records: %u (%llu bytes) at draw heap view %u, made in %.3f ms", x2w_startup::heap_raw_page_records, rd.Width,
            x2w_startup::heap_raw_pages_view, heap_raw_ms(c1.QuadPart - c0.QuadPart));
    }
    for (std::size_t i = 0; i < heap_raw_page_freed.size();)  // runs past their quarantine are free again
        if (frame_number - heap_raw_page_freed[i].frame >= heap_raw_page_quarantine) {
            heap_raw_page_free.emplace(heap_raw_page_freed[i].count, heap_raw_page_freed[i].first);
            heap_raw_page_freed[i] = heap_raw_page_freed.back();
            heap_raw_page_freed.pop_back();
        } else {
            ++i;
        }
    if (const auto run = heap_raw_page_free.lower_bound(count); run != heap_raw_page_free.end()) {
        const std::uint32_t first = run->second, left = run->first - count;
        heap_raw_page_free.erase(run);
        if (left) heap_raw_page_free.emplace(left, first + count);
        return first;
    }
    if (x2w_startup::heap_raw_page_records - heap_raw_page_top < count) return UINT_MAX;
    heap_raw_page_top += count;
    return heap_raw_page_top - count;
}
void heap_raw_page_release(std::uint32_t first, std::uint32_t count) { heap_raw_page_freed.push_back({frame_number, first, count}); }
// Writes records [first, first + records.size() / 2) (shadow and GPU copy, fenced: the entry that names them is
// written after).
void heap_raw_page_store(std::uint32_t first, const std::vector<std::uint32_t>& records)
{
    for (std::size_t i = 0; i < records.size(); ++i) {
        heap_raw_page_shadow[std::size_t{first} * 2 + i] = records[i];
        heap_raw_page_words[std::size_t{first} * 2 + i] = records[i];
    }
    _mm_sfence();
}
// The page records of [base, base + bytes): every 64 KiB page in a window, with room in its view for the range's
// bytes in that page. nullptr, or the counter of why not.
LONG64* heap_raw_page_runs(u64 base, std::uint32_t bytes, std::vector<std::uint32_t>& records)
{
    records.clear();
    const std::uint32_t count = heap_raw_page_count(base, bytes);
    if (count > x2w_startup::heap_raw_paged_max_pages) return &heap_raw_unmapped;
    const u64 first = base >> 16, end = base + bytes;
    for (std::uint32_t i = 0; i < count; ++i) {
        const HeapRawPage& p = heap_raw_page(first + i);
        if (!p.window) return &heap_raw_unmapped;
        u64 view_bytes = 0;
        const UINT index = heap_raw_window(p.window, &view_bytes);
        if (index == UINT_MAX) return &heap_raw_no_window;
        const u64 page_start = (first + i) << 16, used = (std::min)(end, page_start + 0x10000) - page_start;
        if (p.offset + used > view_bytes || (p.offset & 3) || p.offset > 0xffffffffull) return &heap_raw_past_view;
        records.push_back(index);
        records.push_back(static_cast<std::uint32_t>(p.offset));
    }
    return nullptr;
}

// Resolves a range: its window field and offset, or the counter of why not. One window: the window (with
// heap_raw_window_unaligned for a base that is not 4-aligned; the hardware reads such raw buffers byte-exact, and so
// does the program, two dwords per load). Else, every page in some window: a paged entry (new page records).
LONG64* heap_raw_locate_window(u64 base, std::uint32_t bytes, UINT* index, u64* offset);
LONG64* heap_raw_locate(u64 base, std::uint32_t bytes, UINT* index, u64* offset)
{
    LONG64* why = heap_raw_locate_window(base, bytes, index, offset);
    if (why != &heap_raw_unmapped) return why;
    static std::vector<std::uint32_t> records;
    if (LONG64* not_paged = heap_raw_page_runs(base, bytes, records)) return not_paged;
    const std::uint32_t first = heap_raw_page_alloc(static_cast<std::uint32_t>(records.size() / 2));
    if (first == UINT_MAX) return &heap_raw_paged_full;
    heap_raw_page_store(first, records);
    *index = x2w_startup::heap_raw_window_paged;
    *offset = first;
    if (!heap_raw_revalidating) ++heap_raw_paged;
    static LONG shown;
    if (first_calls(shown, 8))
        log("x2w-heapraw paged range: base 0x%llx bytes %u, %zu pages, records %u.. (first page window %u offset 0x%x)", base, bytes,
            records.size() / 2, first, records[0], records[1]);
    return nullptr;
}

// The one-window part of heap_raw_locate.
LONG64* heap_raw_locate_window(u64 base, std::uint32_t bytes, UINT* index, u64* offset)
{
    const u64 first = base >> 16, last = (base + bytes - 1) >> 16;
    const HeapRawPage p0 = heap_raw_page(first);
    ID3D12Resource* window = p0.window;
    for (u64 page = first + 1; window && page <= last; ++page) {
        const HeapRawPage& p = heap_raw_page(page);
        if (p.window != window || p.offset != p0.offset + ((page - first) << 16)) window = nullptr;
    }
    *offset = p0.offset + (base & 0xffff);
    if (!window && !desktop_range(base, bytes, &window, offset)) window = nullptr;  // the page path refused: ask for the range itself
    else if (window && heap_raw_check() && !heap_raw_revalidating) {
        ID3D12Resource* w2 = nullptr;
        u64 o2 = 0;
        const bool ok2 = desktop_range(base, bytes, &w2, &o2);
        if (ok2 != (window != nullptr) || (ok2 && (w2 != window || o2 != *offset))) {
            static LONG shown;
            if (first_calls(shown, 8))
                log("x2w-heapraw check: base 0x%llx bytes %u: pages %p+0x%llx, desktop_range %d %p+0x%llx", base, bytes, static_cast<void*>(window),
                    *offset, ok2, static_cast<void*>(w2), o2);
            ++heap_raw_check_mismatch;
        }
    }
    if (!window) return &heap_raw_unmapped;
    u64 view_bytes = 0;
    if ((*index = heap_raw_window(window, &view_bytes)) == UINT_MAX) return &heap_raw_no_window;
    if (*offset + bytes > view_bytes || (*offset & 3) != (base & 3) || *offset > 0xffffffffull) return &heap_raw_past_view;
    if (base & 3) {
        *index |= x2w_startup::heap_raw_window_unaligned;
        if (!heap_raw_revalidating) ++heap_raw_unaligned_resolved;
    }
    return nullptr;
}

// One sighting of (base, bytes): refreshed if in the table (a refused one retried when due), else resolved and
// inserted. True when it needed a resolution attempt.
bool heap_raw_take(u64 base, std::uint32_t bytes)
{
    std::uint32_t place = UINT_MAX;
    std::uint32_t at = heap_raw_find(base, bytes, &place);
    if (at != UINT_MAX) {
        HeapRawSlot& s = heap_raw_shadow[at];
        s.seen = heap_raw_pass;
        if (s.window() != heap_raw_refused || frame_number - s.tried < static_cast<LONG>(x2w_startup::heap_raw_retry_frames)) return false;
    }
    UINT index = heap_raw_refused;
    u64 offset = 0;
    LONG64* refused = heap_raw_locate(base, bytes, &index, &offset);
    if (refused) {
        ++*refused;  // draw_lock held (as every counter here but noted / dropped)
        index = heap_raw_refused;
        offset = 0;
    } else {
        ++heap_raw_resolved;
        ++heap_raw_hb_resolved;
        static LONG shown;
        if (first_calls(shown, 8))
            log("x2w-heapraw resolved V# base 0x%llx bytes %u -> window %u offset 0x%llx", base, bytes, index, offset);
    }
    if (at != UINT_MAX) {  // a refused range, tried again
        if (!refused) {
            heap_raw_write(at, base, bytes, index, static_cast<std::uint32_t>(offset));
            heap_raw_shadow[at].checked = heap_raw_pass;
            --heap_raw_refused_live;
        }
        heap_raw_shadow[at].tried = frame_number;
        return true;
    }
    HeapRawSlot& old = heap_raw_shadow[place];
    if (old.bytes && old.window() != heap_raw_tombstone) {  // every probe occupied: the least recently seen goes
        ++heap_raw_displaced;
        ++heap_raw_hb_displaced;
        if (heap_raw_seen(place) + 1 >= heap_raw_pass) ++heap_raw_displaced_recent;
        if (old.window() == x2w_startup::heap_raw_window_paged) heap_raw_page_release(old.offset, heap_raw_page_count(old.base(), old.bytes));
        if (old.window() == heap_raw_refused) --heap_raw_refused_live;
        --heap_raw_live;
    }
    heap_raw_write(place, base, bytes, index, static_cast<std::uint32_t>(offset));
    heap_raw_shadow[place].seen = heap_raw_pass;
    heap_raw_shadow[place].checked = heap_raw_pass;
    heap_raw_shadow[place].tried = frame_number;
    if (!heap_raw_bits.empty()) heap_raw_bits[place >> 6] &= ~(1ull << (place & 63));  // the bit was the old entry's
    ++heap_raw_live;
    if (refused) ++heap_raw_refused_live;
    return true;
}

// Reads the whole title heap (draw_lock held; the first drain and a rescan): every V# of the runtime's shape goes in
// `out`. Returns the V#s found; a heap the CPU cannot read is skipped (logged).
std::size_t heap_raw_read_all(u64 heap, u64 slots, std::vector<HeapRawPending>& out)
{
    if (!x2w_local::readable(reinterpret_cast<const void*>(heap), static_cast<std::size_t>(slots * 32))) {
        log("x2w-heapraw heap 0x%llx (%llu slots) not readable, skipped", heap, slots);
        return 0;
    }
    std::size_t found = 0;
    u64 last_base = 0;
    std::uint32_t last_bytes = 0;
    for (u64 i = 0; i < slots; ++i) {
        std::uint32_t w[8];
        std::memcpy(w, reinterpret_cast<const void*>(heap + i * 32), sizeof w);
        x2w_startup::RawBufferWords v;
        if (!x2w_startup::runtime_vdesc_shape(w) || !x2w_startup::raw_buffer_words(w, v)) continue;
        ++found;
        if (v.base == last_base && v.bytes == last_bytes) continue;  // the slot before held the same range
        last_base = v.base;
        last_bytes = v.bytes;
        out.push_back({v.base, v.bytes});
    }
    return found;
}

bool heap_raw_sweep_sync()
{
    static const bool on = heap_raw_env("X2W_HEAP_RAW_SWEEP_SYNC");
    return on;
}
struct HeapRawSweep {
    u64 heap = 0, from = 0, to = 0, slots = 0;  // the part asked for
    std::uint32_t pass = 0;
    std::vector<std::uint64_t> building, done;  // bits of the pass being read, and of the last one read to the end
    std::uint32_t done_pass = 0;
    bool pass_done = false;                     // `done` is new
    std::vector<HeapRawPending> unknown;
    std::size_t found = 0;                      // V#s in the part
    LONG64 ticks = 0;                           // the part's time on the thread
    LONG64 faults = 0;                          // 256-slot blocks that faulted (unmapped while read), skipped
};
// Copies title memory on the sweep thread: false when a page faults (the heap unmapped under the read).
// Page by page: a guard page's fault clears its guard bit, which is set again (as x2w_local::readable does).
bool heap_raw_copy(void* to, const void* from, std::size_t bytes)
{
    const auto start = reinterpret_cast<std::uintptr_t>(from);
    for (std::uintptr_t at = start; at < start + bytes;) {
        const std::uintptr_t next = (std::min)((at | 0xFFF) + 1, start + bytes);
        DWORD code = 0;
        __try {
            std::memcpy(static_cast<unsigned char*>(to) + (at - start), reinterpret_cast<const void*>(at), next - at);
        } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
            if (code == 0x80000001u) {  // STATUS_GUARD_PAGE_VIOLATION
                MEMORY_BASIC_INFORMATION info{};
                DWORD old = 0;
                if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) && info.State == MEM_COMMIT)
                    VirtualProtect(reinterpret_cast<void*>(at & ~std::uintptr_t{0xFFF}), 0x1000, info.Protect | PAGE_GUARD, &old);
            }
            return false;
        }
        at = next;
    }
    return true;
}
HeapRawSweep heap_raw_sweep;
volatile LONG heap_raw_sweep_busy;
HANDLE heap_raw_sweep_go, heap_raw_sweep_done;
LONG64 heap_raw_sweep_behind, heap_raw_sweep_parts, heap_raw_hb_thread_ticks;

// heap_raw_find without a lock and without an insert place (the sweep thread).
std::uint32_t heap_raw_peek(u64 base, std::uint32_t bytes)
{
    const std::uint32_t mask = heap_raw_entries() - 1, lo = static_cast<std::uint32_t>(base), hi = static_cast<std::uint32_t>((base >> 32) & 0xffff);
    const std::uint32_t h = x2w_startup::raw_map_hash(lo, hi, bytes);
    const volatile HeapRawSlot* shadow = heap_raw_shadow.data();
    for (std::uint32_t i = 0; i < x2w_startup::heap_raw_probes; ++i) {
        const volatile HeapRawSlot& e = shadow[(h + i) & mask];
        const std::uint32_t z = e.bytes, y = e.hi_window, x = e.lo;
        if (!z) break;
        if ((y >> 16) == heap_raw_tombstone) continue;
        if (x == lo && (y & 0xffff) == hi && z == bytes) return (h + i) & mask;
    }
    return UINT_MAX;
}

DWORD WINAPI heap_raw_sweep_thread(void*)
{
    for (;;) {
        WaitForSingleObject(heap_raw_sweep_go, INFINITE);
        HeapRawSweep& w = heap_raw_sweep;
        LARGE_INTEGER t0, t1;
        QueryPerformanceCounter(&t0);
        if (w.from == 0) std::fill(w.building.begin(), w.building.end(), 0);
        w.found = 0;
        u64 last_base = 0;
        std::uint32_t last_bytes = 0, last_at = UINT_MAX;
        static std::uint32_t block[256 * 8];  // 256 slots, copied at once (a fault skips the block)
        for (u64 b = w.from; b < w.to; b += 256) {
            const u64 n = (std::min<u64>)(256, w.to - b);
            if (!heap_raw_copy(block, reinterpret_cast<const void*>(w.heap + b * 32), static_cast<std::size_t>(n * 32))) {
                ++w.faults;
                continue;
            }
            for (u64 i = 0; i < n; ++i) {
                const std::uint32_t* words = block + i * 8;
                x2w_startup::RawBufferWords v;
                if (!x2w_startup::runtime_vdesc_shape(words) || !x2w_startup::raw_buffer_words(words, v)) continue;
                ++w.found;
                if (v.base == last_base && v.bytes == last_bytes && last_at != UINT_MAX) continue;
                last_base = v.base;
                last_bytes = v.bytes;
                last_at = heap_raw_peek(v.base, v.bytes);
                if (last_at != UINT_MAX) w.building[last_at >> 6] |= 1ull << (last_at & 63);
                else if (w.unknown.size() < heap_raw_backlog_max) w.unknown.push_back({v.base, v.bytes});
            }
        }
        if (w.to >= w.slots) {
            w.done.swap(w.building);
            std::fill(w.building.begin(), w.building.end(), 0);  // the next pass starts with no sightings
            w.done_pass = w.pass;
            w.pass_done = true;
        }
        QueryPerformanceCounter(&t1);
        w.ticks = t1.QuadPart - t0.QuadPart;
        InterlockedExchange(&heap_raw_sweep_busy, 0);
        SetEvent(heap_raw_sweep_done);
    }
}

// The thread's results, when it is idle (draw_lock held).
void heap_raw_sweep_take()
{
    if (InterlockedCompareExchange(&heap_raw_sweep_busy, 0, 0)) return;
    HeapRawSweep& w = heap_raw_sweep;
    for (const HeapRawPending& p : w.unknown)
        if (heap_raw_backlog.size() < heap_raw_backlog_max) heap_raw_backlog.push_back(p);
    w.unknown.clear();
    heap_raw_hb_thread_ticks += w.ticks;
    w.ticks = 0;
    if (w.pass_done) {
        w.pass_done = false;
        heap_raw_bits.swap(w.done);  // the thread's old `done` comes back as the next one to fill
        heap_raw_bits_pass = w.done_pass;
        heap_raw_pass = w.done_pass + 1;
        ++heap_raw_passes;
    }
}

// Once per frame: take the sweep's results, ask for the next part, then the eviction clock hand (with revalidation).
u64 heap_raw_sweep_heap, heap_raw_sweep_at;
std::uint32_t heap_raw_hand;
LONG heap_raw_swept_frame = -1;
LONG64 heap_raw_revalidated, heap_raw_revalidate_changed;
std::uint32_t heap_raw_revalidate_passes()
{
    static const std::uint32_t n = [] {
        char v[8] = "";
        const DWORD got = GetEnvironmentVariableA("X2W_HEAP_RAW_REVALIDATE_PASSES", v, sizeof v);
        const unsigned long t = got && got < sizeof v ? std::strtoul(v, nullptr, 10) : 0;
        return t >= 1 && t <= 1000 ? static_cast<std::uint32_t>(t) : x2w_startup::heap_raw_revalidate_passes;
    }();
    return n;
}
void heap_raw_sweep_ask(u64 heap, u64 slots, std::uint32_t cap);
void heap_raw_hand_work(std::uint32_t cap, LONG64* evicted_out);
void heap_raw_frame_work(u64 heap, u64 heap_bytes)
{
    if (heap_raw_swept_frame == frame_number) return;
    heap_raw_swept_frame = frame_number;
    LARGE_INTEGER t0, t1;
    QueryPerformanceCounter(&t0);
    const u64 slots = heap_bytes / 32;
    const std::uint32_t cap = heap_raw_entries();
    if (!heap_raw_sweep_go) {
        heap_raw_sweep_go = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        heap_raw_sweep_done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        heap_raw_sweep.building.assign(cap / 64 + 1, 0);
        heap_raw_sweep.done.assign(cap / 64 + 1, 0);
        heap_raw_bits.assign(cap / 64 + 1, 0);
        HANDLE t = CreateThread(nullptr, 0, heap_raw_sweep_thread, nullptr, 0, nullptr);
        if (t) CloseHandle(t);
        log("x2w-heapraw sweep thread %s%s", t ? "started" : "NOT started (no sweep)", heap_raw_sweep_sync() ? " (X2W_HEAP_RAW_SWEEP_SYNC=1)" : "");
        if (!t) heap_raw_sweep_busy = 1;  // never idle: no parts are asked for
    }
    // The order: take the thread's results; then (thread idle) the clock hand, which also counts the sightings of the
    // pass in progress (heap_raw_sweep.building); then ask for the next part. An entry found by a part already read
    // is never evicted on the last complete pass alone (a late staging copy of an entry that pass missed).
    // X2W_HEAP_RAW_SWEEP_SYNC=1 asks and waits before the hand instead.
    heap_raw_sweep_take();
    const bool sync = heap_raw_sweep_sync();
    if (sync) heap_raw_sweep_ask(heap, slots, cap);
    const bool idle = !InterlockedCompareExchange(&heap_raw_sweep_busy, 0, 0);
    if (!idle) ++heap_raw_sweep_behind;
    LONG64 evicted = 0;
    if (idle && heap_raw_live) heap_raw_hand_work(cap, &evicted);
    if (!sync && idle) heap_raw_sweep_ask(heap, slots, cap);
    heap_raw_evicted += evicted;
    heap_raw_hb_evicted += evicted;
    QueryPerformanceCounter(&t1);
    heap_raw_hb_sweep_ticks += t1.QuadPart - t0.QuadPart;
    heap_raw_hb_sweep_max_ticks = (std::max)(heap_raw_hb_sweep_max_ticks, t1.QuadPart - t0.QuadPart);
    if (heap_raw_times())
        log("x2w-heapraw frame work: %.3f ms, sweep at %llu of %llu slots, pass %u, %lld evicted, %u entries, revalidated %lld (changed %lld), "
            "sweep behind %lld, thread %.3f ms",
            heap_raw_ms(t1.QuadPart - t0.QuadPart), heap_raw_sweep_at, slots, heap_raw_pass, evicted, heap_raw_live, heap_raw_revalidated,
            heap_raw_revalidate_changed, heap_raw_sweep_behind, heap_raw_ms(heap_raw_sweep.ticks));
}

// The next part of the heap for the sweep thread (idle). Sync: waited for and taken.
void heap_raw_sweep_ask(u64 heap, u64 slots, std::uint32_t cap)
{
    if (InterlockedCompareExchange(&heap_raw_sweep_busy, 0, 0)) return;
    if (slots) {
        if (heap != heap_raw_sweep_heap) {  // a new heap: the sweep starts again (the pass does not end early)
            if (heap_raw_sweep_heap) log("x2w-heapraw sweep: heap 0x%llx -> 0x%llx (%llu slots)", heap_raw_sweep_heap, heap, slots);
            heap_raw_sweep_heap = heap;
            heap_raw_sweep_at = 0;
        }
        const u64 frames = heap_raw_live > cap / 2 ? 4 : x2w_startup::heap_raw_sweep_frames;
        const u64 chunk = (std::max<u64>)(4096, (slots + frames - 1) / frames);
        HeapRawSweep& w = heap_raw_sweep;
        w.heap = heap;
        w.slots = slots;
        w.from = heap_raw_sweep_at;
        w.to = (std::min)(slots, heap_raw_sweep_at + chunk);
        w.pass = heap_raw_pass;
        heap_raw_sweep_at = w.to >= slots ? 0 : w.to;
        ++heap_raw_sweep_parts;
        InterlockedExchange(&heap_raw_sweep_busy, 1);
        SetEvent(heap_raw_sweep_go);
        if (heap_raw_sweep_sync()) {
            WaitForSingleObject(heap_raw_sweep_done, 2000);
            heap_raw_sweep_take();
        }
    }
}

constexpr LONG64 heap_raw_revalidate_pages = 256;
void heap_raw_hand_work(std::uint32_t cap, LONG64* evicted_out)
{
    LONG64 evicted = 0;
    std::uint32_t revalidate = 1024;  // re-resolutions per frame at most
    const LONG64 lookups_end = heap_raw_page_lookups + heap_raw_revalidate_pages;
    const std::uint32_t every = heap_raw_revalidate_passes();
    const std::vector<std::uint64_t>& now = heap_raw_sweep.building;  // the pass in progress, so far
    heap_raw_revalidating = true;
    {
        // 1/16 of the table per frame, 1/4 when past half full (the sweep is 4x faster then too), at least 4096 entries
        const bool evict = heap_raw_live > cap / 4;
        for (std::uint32_t n = (std::max)((std::min)(cap, 4096u), heap_raw_live > cap / 2 ? cap / 4 : cap / 16); n; --n, heap_raw_hand = (heap_raw_hand + 1) & (cap - 1)) {
            HeapRawSlot& e = heap_raw_shadow[heap_raw_hand];
            if (!e.bytes || e.window() == heap_raw_tombstone) continue;
            const bool seen_now = (now[heap_raw_hand >> 6] >> (heap_raw_hand & 63)) & 1;
            if (evict && !seen_now && heap_raw_seen(heap_raw_hand) + 2 <= heap_raw_pass) {
                if (e.window() == heap_raw_refused) --heap_raw_refused_live;
                if (e.window() == x2w_startup::heap_raw_window_paged) heap_raw_page_release(e.offset, heap_raw_page_count(e.base(), e.bytes));
                heap_raw_set_window(heap_raw_hand, heap_raw_tombstone);
                --heap_raw_live;
                ++evicted;
                continue;
            }
            // Remap healing: a resolved entry is resolved again every `every` passes (the window and offset of its
            // pages now); a changed one is rewritten in place, one that no longer resolves becomes refused.
            if (revalidate && heap_raw_page_lookups < lookups_end && e.window() != heap_raw_refused && e.checked + every <= heap_raw_pass) {
                --revalidate;
                ++heap_raw_revalidated;
                e.checked = heap_raw_pass;
                UINT index = heap_raw_refused;
                u64 offset = 0;
                const u64 base = e.base();
                const std::uint32_t bytes = e.bytes;
                if (e.window() == x2w_startup::heap_raw_window_paged) {  // its page records again, rewritten in place when changed
                    static std::vector<std::uint32_t> records;
                    const std::uint32_t first = e.offset;
                    if (LONG64* refused = heap_raw_page_runs(base, bytes, records)) {
                        ++*refused;
                        ++heap_raw_revalidate_changed;
                        heap_raw_page_release(first, heap_raw_page_count(base, bytes));
                        heap_raw_write(heap_raw_hand, base, bytes, heap_raw_refused, 0);
                        e.tried = frame_number;
                        ++heap_raw_refused_live;
                    } else if (!std::equal(records.begin(), records.end(), heap_raw_page_shadow.begin() + std::size_t{first} * 2)) {
                        ++heap_raw_revalidate_changed;
                        heap_raw_page_store(first, records);
                    }
                    continue;
                }
                if (LONG64* refused = heap_raw_locate(base, bytes, &index, &offset)) {
                    ++*refused;
                    ++heap_raw_revalidate_changed;
                    heap_raw_write(heap_raw_hand, base, bytes, heap_raw_refused, 0);
                    e.tried = frame_number;
                    ++heap_raw_refused_live;
                } else if (index != e.window() || offset != e.offset) {
                    ++heap_raw_revalidate_changed;
                    heap_raw_write(heap_raw_hand, base, bytes, index, static_cast<std::uint32_t>(offset));
                }
            }
        }
        heap_raw_flush();
        _mm_sfence();
    }
    heap_raw_revalidating = false;
    *evicted_out = evicted;
}

// A `heapraw 1` dispatch, draw_lock held. Every range noted since the last drain is looked up, and the new ones are
// resolved now (this frame's dispatches may read them). The first drain reads the whole heap into the backlog; the
// sweep (once per frame) adds what it finds; at most heap_raw_resolve_budget backlog ranges are resolved per drain.
// False: no table.
bool heap_raw_drain(u64 heap, u64 heap_bytes)
{
    if (!heap_raw_on() || !heap_raw_ensure()) return false;
    LARGE_INTEGER t0, t1;
    QueryPerformanceCounter(&t0);
    std::vector<HeapRawPending> pending;
    AcquireSRWLockExclusive(&heap_raw_pending_lock);
    pending.swap(heap_raw_pending);
    const bool rescan = heap_raw_rescan != 0;
    heap_raw_rescan = 0;
    ReleaseSRWLockExclusive(&heap_raw_pending_lock);
    if ((!heap_raw_scans || rescan) && heap && heap_bytes >= 32) {
        // The whole heap, resolved now with the noted ranges (no budget): every V# it holds is in the table before
        // this dispatch reads it.
        ++heap_raw_scans;
        const std::size_t noted = pending.size();
        const std::size_t found = heap_raw_read_all(heap, heap_bytes / 32, pending);
        log("x2w-heapraw heap read %lld (%s): %llu slots of heap 0x%llx, %zu V#s of the runtime's shape, %zu ranges with %zu noted, all resolved "
            "now", heap_raw_scans, rescan ? "rescan, the pending list was full" : "first drain", heap_bytes / 32, heap, found, pending.size(), noted);
    }
    const LONG64 before = heap_raw_resolved + heap_raw_unaligned + heap_raw_unmapped + heap_raw_past_view + heap_raw_no_window + heap_raw_paged_full;
    const std::uint32_t mask = heap_raw_entries() - 1;
    for (std::size_t i = 0; i < pending.size(); ++i) {
        if (i + 8 < pending.size()) {  // the shadow line of a range 8 ahead (a cache miss each otherwise)
            const HeapRawPending& q = pending[i + 8];
            const std::uint32_t h = x2w_startup::raw_map_hash(static_cast<std::uint32_t>(q.base), static_cast<std::uint32_t>((q.base >> 32) & 0xffff), q.bytes);
            _mm_prefetch(reinterpret_cast<const char*>(&heap_raw_shadow[h & mask]), _MM_HINT_T0);
        }
        heap_raw_take(pending[i].base, pending[i].bytes);
    }
    heap_raw_flush();
    heap_raw_frame_work(heap, heap_bytes);
    if (!heap_raw_backlog.empty()) {
        std::uint32_t budget = x2w_startup::heap_raw_resolve_budget;
        std::size_t i = 0;
        for (; i < heap_raw_backlog.size() && budget; ++i) budget -= heap_raw_take(heap_raw_backlog[i].base, heap_raw_backlog[i].bytes) ? 1 : 0;
        heap_raw_deferred += heap_raw_backlog.size() - i;
        heap_raw_backlog.erase(heap_raw_backlog.begin(), heap_raw_backlog.begin() + i);
        heap_raw_flush();
    }
    QueryPerformanceCounter(&t1);
    const LONG64 after = heap_raw_resolved + heap_raw_unaligned + heap_raw_unmapped + heap_raw_past_view + heap_raw_no_window + heap_raw_paged_full;
    if (pending.empty() && after == before) return true;  // no work
    ++heap_raw_hb_drains;
    heap_raw_hb_ticks += t1.QuadPart - t0.QuadPart;
    heap_raw_hb_max_ticks = (std::max)(heap_raw_hb_max_ticks, t1.QuadPart - t0.QuadPart);
    if (heap_raw_times())
        log("x2w-heapraw drain time: %.3f ms, %zu noted, %lld outcomes, %zu waiting, %lld page lookups, %u entries, evicted %lld, displaced %lld "
            "(recent %lld), check mismatches %lld",
            heap_raw_ms(t1.QuadPart - t0.QuadPart), pending.size(), after - before, heap_raw_backlog.size(), heap_raw_page_lookups, heap_raw_live,
            heap_raw_evicted, heap_raw_displaced, heap_raw_displaced_recent, heap_raw_check_mismatch);
    LONG64 power = 1;  // a counter line when the outcomes total crosses a power of two
    while (power * 2 <= after) power *= 2;
    if (after && power > before) heap_raw_counters();
    return true;
}

// The heartbeat (every 5 s, d3d12_xs_device.cpp), for the frame-time gate of a watched run. There is no whole-table
// reset any more; "table half full: reset" counts the entries evicted and displaced instead.
// Called from Present outside draw_lock (the heartbeat block runs after the lock_draw section above it is released).
void report_heap_raw()
{
    if (!heap_raw_on() || (!heap_raw_map && !heap_raw_dropped)) return;
    AcquireSRWLockExclusive(&draw_lock);
    const LONG64 drains = heap_raw_hb_drains, ticks = heap_raw_hb_ticks, max_ticks = heap_raw_hb_max_ticks, resolved = heap_raw_hb_resolved,
                 evicted = heap_raw_hb_evicted, sweep = heap_raw_hb_sweep_ticks, sweep_max = heap_raw_hb_sweep_max_ticks,
                 displaced = heap_raw_hb_displaced;
    heap_raw_hb_drains = heap_raw_hb_ticks = heap_raw_hb_max_ticks = heap_raw_hb_resolved = heap_raw_hb_evicted = heap_raw_hb_sweep_ticks =
        heap_raw_hb_sweep_max_ticks = heap_raw_hb_displaced = 0;
    const std::size_t waiting = heap_raw_backlog.size();
    static LONG64 dropped_reported;
    const LONG64 dropped = heap_raw_dropped - dropped_reported;
    dropped_reported = heap_raw_dropped;
    const std::uint32_t live = heap_raw_live, refused = heap_raw_refused_live;
    const LONG64 passes = heap_raw_passes, thread = heap_raw_hb_thread_ticks, behind = heap_raw_sweep_behind, faults = heap_raw_sweep.faults;
    heap_raw_hb_thread_ticks = 0;
    ReleaseSRWLockExclusive(&draw_lock);
    // Drain times include the sweep and eviction work that ran inside them.
    log("x2w-heapraw drains with work: %lld, %.3f ms each, max %.3f ms (last 5 s, draw_lock held); table half full: reset 0 (evicted %lld, "
        "displaced %lld); resolved %lld new ranges, %zu waiting, %lld notes dropped (pending full); %u of %u entries (%u refused); sweep "
        "max %.3f ms per frame (%.3f ms in all), %lld passes; sweep thread %.3f ms (off draw_lock), behind %lld, faulted blocks %lld",
        drains, drains ? heap_raw_ms(ticks) / static_cast<double>(drains) : 0.0, heap_raw_ms(max_ticks), evicted, displaced, resolved, waiting,
        dropped, live, heap_raw_entries(), refused, heap_raw_ms(sweep_max), heap_raw_ms(sweep), passes,
        heap_raw_ms(thread), behind, faults);
}

// The table's view (t2 space1) at `dest`: a raw SRV of the table, or a null raw view (switch off / no table).
void heap_raw_bind(bool table, D3D12_CPU_DESCRIPTOR_HANDLE dest)
{
    if (table && heap_raw_map) {
        bounded_raw_srv(heap_raw_map, 0, u64{heap_raw_entries()} * 16, dest);
        return;
    }
    bounded_raw_srv(nullptr, 0, 4, dest);  // a null view: GetDimensions reads 0
}
