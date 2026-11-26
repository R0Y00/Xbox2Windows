// Experimental (docs/startup-runtime.md).
//
// With X2W_D3D12_PERMISSIVE=1, D3D12XboxCreateDevice accepts a non-null Xbox parameter block and
// returns a recording wrapper around a desktop ID3D12Device. Every object the wrappers hand out is
// wrapped too. Every vtable slot traps ("x2w-trap d3d12_xs.dll!<Object>::slot<N>", then terminates)
// unless it is enabled in build_tables(). Enabled slots log "x2w-d3d12" lines (the first calls only).
//
// Evidence so far: the title requests desktop IIDs (ID3D12Device11, ID3D12CommandQueue, ID3D12Fence,
// ID3D12CommandAllocator), and device slots 8, 9, 14, 26, 28 and 36 have desktop call shapes. Slot 48
// does not (Xbox queue creation), so slots from 48 up are only enabled one by one from call sites.
// Methods that return a structure by value are called with the hidden result pointer in rcx and the
// object in rdx (desktop: object first); those slots accept either order.
#define X2W_TRAP_MODULE "d3d12_xs.dll"
#define X2W_TRAP_MODULE_ID 7
#include "startup_stubs/trap.h"
#include "x2w_build_rev.h"
#include "xmem_page_batch.h"
#include "startup_stubs/local_log.h"
#include "xmem_low.h"
#include "startup_stubs/frame_rgb_sample.h"

#include <algorithm>
#include <deque>
#include <map>
#include <tuple>
#include <d3d12.h>
#include "startup_stubs/logical_view_table.h"
#include "startup_stubs/list_compute_root.h"
#include "startup_stubs/placed_pool.h"
#include "startup_stubs/small_texture_heap.h"
#include "startup_stubs/overlay_window.h"
#include "startup_stubs/title_composition.h"
#include "startup_stubs/draw_bmp.h"
#include "startup_stubs/draw_dump_tracking.h"
#include "startup_stubs/translated_sampler_bindings.h"
#include "startup_stubs/export_write_mask.h"
#include "startup_stubs/depth_state_apply.h"
#include "startup_stubs/d16_image_format.h"
#include "startup_stubs/formatted_buffer.h"
#include "startup_stubs/heap_indexed_root.h"
#include "startup_stubs/indexed_indirect_guard.h"
#include "startup_stubs/indexed_rect_geometry.h"
#include "startup_stubs/bc7_4k_s.h"
#include "startup_stubs/bc64_chain.h"
#include "startup_stubs/gpu_resident_mips.h"
#include "startup_stubs/gfx10_layout_local.h"  // LOCAL (textures): X2W_MEMORY_SRV_DETILE (x2w_startup::gfx10_local)
#include "startup_stubs/buffer_address_index.h"  // LOCAL (textures): existing_memory_buffer's address index
#include "startup_stubs/address_texture.h"
#include "startup_stubs/list_upload_pages.h"
#include "startup_stubs/resource_budget.h"
#include <optional>
#include "startup_stubs/derived_blend.h"
#include "startup_stubs/placement_publish.h"
#include "startup_stubs/resource_address_index.h"
#include "startup_stubs/placed_build.h"
#include "startup_stubs/placed_buffer_range.h"
#include "startup_stubs/memory_srv_report.h"
#include "startup_stubs/memory_srv_census.h"
#include "startup_stubs/memory_srv_as.h"
#include "startup_stubs/per_reason_refusal_census.h"
#include "startup_stubs/buffer_format_word3.h"
#include "startup_stubs/memory_srv_buffer.h"
#include "startup_stubs/single_element_srv.h"
#include "startup_stubs/list_copy.h"
#include "startup_stubs/d3d12_teardown.h"
#include "startup_stubs/rect_indirect_args.h"
#include "startup_stubs/profile_zone.h"

#include "startup_stubs/slice_footprints.h"
#include "startup_stubs/present_signature_sampler.h"

#include <d3d12.h>
#include "command_list_slot_names.h"
#include "command_signature_filter.h"
#include "console_plane_size.h"
#include "descriptor_heap_type.h"
#include "small_texture_4k.h"
#include "null_binding_cause.h"
#include "null_texture_view.h"
#include "packed_target_format.h"
#include "gpu_srgb.h"
#include "gpu_read_states.h"
#include "gpu_build_queue.h"
#include "gpu_warm_bad.h"
#include "gpu_pipeline_record.h"
#include "gpu_reference_library.h"
#include "startup_stubs/gpu_trace_request.h"
#include "startup_stubs/gpu_transition.h"
#include <filesystem>
#include <fstream>
#include "present_view_format.h"
#include "state_block_blend.h"
#include "title_heap_memory.h"
#include "gfx10_view_words.h"
#include "gfx10_raw_vdesc.h"
#include <xmmintrin.h>  // _mm_sfence (d3d12_heap_raw.inl: the upload-heap table)
#include "startup_stubs/rawaddr_table.h"  // LOCAL raw-address shaders (d3d12_rawaddr.inl)
#include "startup_stubs/d3d12_mem_counters.h"  // LOCAL memcut counters
#include "xmem_section_lease.h"
#include "startup_stubs/xmem_section_buffer.h"
// Translated shader ABI (X2W_GPU_ABI=1, PLAN.md WP0-WP5): the standalone headers; the hook prototypes follow
// struct Wrapper.
#include "startup_stubs/gpu_abi.h"
#include "startup_stubs/gpu_view_trace.h"
#include "startup_stubs/gpu_db.h"
#include "startup_stubs/gpu_roots.h"
#include "startup_stubs/gpu_list_states.h"
#include "startup_stubs/gpu_pipeline_state.h"
#include "startup_stubs/gpu_heap.h"
#include "startup_stubs/gpu_indirect_clamp.h"
#include <d3d12sdklayers.h>
#include <d3d12shader.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// The heap-indexing bits src/d3d12_xs_shim.cpp removed for the blob it returned at exactly this
// buffer. A build that compiles this file without the shim (offscreen_check) links the fallback, 0.
extern "C" std::uint32_t x2w_stripped_root_flags(const void* blob, std::size_t bytes);
extern "C" std::uint32_t x2w_no_stripped_root_flags(const void*, std::size_t) { return 0; }
#pragma comment(linker, "/alternatename:x2w_stripped_root_flags=x2w_no_stripped_root_flags")

namespace {

using u64 = std::uint64_t;
constexpr int slot_count = 512;
// Refused slot 70 calls, "token" texture nulls (memory_srv_census.h), and indirect refusal censuses
// per (pipeline key, reason) (per_reason_refusal_census.h), printed with the five-second report.
x2w_startup::MemorySrvCensus memory_srv_census;
x2w_startup::TokenNullCensus token_null_census;
x2w_startup::MemorySrvWrites memory_srv_writes;
x2w_startup::PerReasonRefusalCensus indirect_draw_refusal_census{"indirect-draw"};
x2w_startup::PerReasonRefusalCensus indirect_other_refusal_census{"indirect-other"};

enum Kind : int {
    device_kind, queue_kind, fence_kind, allocator_kind, list_kind, descriptor_heap_kind, root_signature_kind,
    resource_kind, heap_kind, query_heap_kind, pipeline_kind, command_signature_kind, other_kind, kind_count
};
const char* const kind_names[kind_count] = {
    "Device", "CommandQueue", "Fence", "CommandAllocator", "CommandList", "DescriptorHeap", "RootSignature",
    "Resource", "Heap", "QueryHeap", "PipelineState", "CommandSignature", "Object"};

// The title reads console object fields directly. Seen on a command list: +0x18 is the command
// buffer cursor and +0x30 its limit; inlined calls write GPU packets at the cursor while it is below
// the limit and otherwise call the virtual method (IASetPrimitiveTopology: slot 20, +0xa0). Both
// stay 0 here, so every inlined call takes its virtual path. The first 0x400 bytes after the vtable
// pointer are a zero-filled stand-in for such fields; the wrapper's own state follows.
//
// The title also writes into those fields (seen: a command list at +0x544), so the wrapper's own
// state lives in front of the object, out of the title's reach, and the zero-filled area behind the
// vtable pointer is sized per kind (64 KiB for command lists).
struct Wrapper;
struct RootParameter;
enum class IndirectCommand { none, dispatch, draw, draw_indexed, dispatch_mesh };
struct IndirectGraphicsPipeline {
    ID3D12RootSignature* root{};
    DXGI_FORMAT target_format{};
    bool mesh{};
    bool cached_mesh{};
    UINT cached_target_count{};
    DXGI_FORMAT cached_targets[8]{};
    DXGI_FORMAT cached_depth{};
    bool cached_depth_writes{};
    const std::vector<RootParameter>* guest_root_params{};
    UINT cached_srv_root = UINT_MAX;
};

// Why a resource is outside the retire class (X2W_D3D12_RELEASE_RETIRED), decided at creation, and why it left it later.
// Both only feed the opt-in X2W_D3D12_RETIRE_REASONS diagnostics. never::slot is the zero value: created by a path that
// does not classify.
enum class RetireNever : std::uint8_t { slot, none, buffer, dimension, msaa, rt_ds_uav, heap_type, heap_flags, title_memory,
                                        aliased, reserved, count };
enum class RetireLeft : std::uint8_t { none, view, view_copy, residency, present, hold_missed, count };

struct alignas(16) WrapperOwn {
    IUnknown* real;
    LONG refs;
    LONG cold_texture_state;  // See d3d12_cold_texture.inl; zero unless the opt-in owns this committed texture.
    Kind kind;
    x2w_budget::Ticket* budget;
    x2w_budget::Ticket* backing_budget;
    void* title_memory;       // Resource: the title-side memory (placement address or hook allocation), or null
    std::uint64_t title_cpu;  // DescriptorHeap: title-owned descriptor memory, or 0
    // Fence: the completed value the title polls through the address from slot 12 (monitorx/mwaitx).
    // Updated when the fence is signaled; GPU work counts as complete at once while lists are no-ops.
    volatile LONG64 completed;
    bool list_open;   // CommandList (render mode): recording between reset and close
    x2w_upload::Arena* list_uploads; // Current recording's allocations, owned by its desktop allocator.
    ID3D12Resource* list_scratch;    // Current recording's rectangle-list argument scratch, owned by its desktop allocator.
    ID3D12Resource* plane_copy_scratch; // Owned by this recording's desktop allocator.
    bool list_close_failed; // Desktop Reset repeats a failed Close result indefinitely.
    unsigned char srgb_uav_format; // Original typed format of a typeless sRGB/UAV backing texture.
    bool tiled_address_srv; // Resource backed by 4 KiB tiled guest memory for translated draws.
    bool linear_memory_srv : 1; // LOCAL (video): texture made by X2W_MEMORY_SRV_LINEAR; 1D/3D ones are uploaded too.
    bool partial_mip_srv : 1;  // detiled late descriptor, possibly only a coarse suffix is resident
    bool detiled_memory_srv : 1; // LOCAL (textures): with linear_memory_srv, its title memory is console-swizzled
                                 // (X2W_MEMORY_SRV_DETILE, gfx10_layout.h) and upload_texture detiles it. A bit
                                 // field next to linear_memory_srv keeps WrapperOwn's size.
    bool retire_when_idle;  // Resource: in the retire class (X2W_D3D12_RELEASE_RETIRED); cleared by any other use. Plain
                            // accesses: a race needs the guest to use a resource across its own final Release.
    bool retire_hold_missed;  // Resource: retire_hold could not record a use (no desktop object or no room), so the
                              // drain must keep it for the process instead of dropping a list's desktop object.
    bool retire_views_stuck;  // Resource: X2W_D3D12_RELEASE_RETIRED_VIEWS could not clear every descriptor slot that
                              // names it, so it stays retained (and is not scanned again).
    std::uint8_t retire_never;  // RetireNever of this resource (set only with X2W_D3D12_RELEASE_RETIRED)
    std::uint8_t retire_left;   // RetireLeft: the first use that took it out of the class
    volatile LONG retire_view_refs;  // Live tracked slot-18 SRV descriptor slots referencing this resource.
    bool retire_has_written_gen;     // Whether written_textures()[title_memory].generation was snapshotted at Release.
    u64 retire_written_gen;          // Snapshotted written_textures() generation for stale-entry cleanup.
    bool list_ready;  // CommandList (render mode): closed with recorded work, not yet submitted
    bool compute_pipeline;   // PipelineState: a desktop compute pipeline
    u64 compute_trace_key;   // Opt-in trace: selected slot 62 container hash
    const void* compute_trace_root;
    LONG compute_trace_dispatches;
    bool root_without_bindings; // RootSignature: decoded and has no parameters or static samplers
    volatile LONG queue_wait_seen; // CommandQueue: publish later signals only after GPU completion.
    void* pass;       // CommandList (render mode): recording state (Pass)
    Wrapper* list_device;    // CommandList: wrapper that created this list.
    Wrapper* signature_device; // CommandSignature: owner of the supported single-command shape
    IndirectCommand indirect_command;
    ID3D12RootSignature* signature_root; // CommandSignature: desktop root of a Dispatch with root constants, or null
    UINT indirect_bytes; // CommandSignature: bytes one command reads (root constants plus the Dispatch record)
    IndirectGraphicsPipeline indirect_graphics;
    u64 mesh86_key{};
    UINT list_node_mask;
    D3D12_QUERY_HEAP_DESC query_heap; // QueryHeap: the desc it was created with (device slot 39)
    // Translated shader ABI (gpu_hooks.h): [0] pipeline WP2 / descriptor heap WP5 / command list WP3, [1] command
    // list WP4.
    void* gpu[2];
};
static_assert(sizeof(WrapperOwn) % 16 == 0, "the title-visible object must stay 16-byte aligned");
struct Wrapper {
    void** vtable;
    std::uint64_t console_fields[0x7f];  // +0x08..+0x400 (more for command lists), zero until the title writes
    WrapperOwn& own() const { return const_cast<WrapperOwn*>(reinterpret_cast<const WrapperOwn*>(this))[-1]; }
};
#include "startup_stubs/gpu_hooks.h"  // Translated shader ABI hook prototypes (inside this anonymous namespace)

void fence_reached(Wrapper* fence, std::uint64_t value) noexcept
{
    LONG64 seen = fence->own().completed;
    while (static_cast<std::uint64_t>(seen) < value || seen == -1) {
        const LONG64 previous = InterlockedCompareExchange64(&fence->own().completed, static_cast<LONG64>(value), seen);
        if (previous == seen) break;
        seen = previous;
    }
}

// Descriptor heaps created over title memory: title CPU handles in [title, title + bytes) map to
// desktop CPU handles at the same offset (the title's increments come from the forwarded desktop
// GetDescriptorHandleIncrementSize, so offsets agree). Descriptors the title writes or copies into
// that memory itself are not seen by the desktop heap.
struct DescriptorRange {
    std::uint64_t title;
    std::uint64_t bytes;
    std::uint64_t desktop;
    UINT title_stride;
    UINT desktop_stride;
    // How many title descriptors the desktop heap covers. Normally the whole heap; shorter only when
    // the console heap was larger than the driver accepts and the stand-in emulated the remainder.
    std::uint64_t desktop_descriptors;
    std::uint64_t tail_cpu; // CPU-only heap for view records outside the shader-visible prefix.
};
SRWLOCK descriptor_lock = SRWLOCK_INIT;
std::vector<DescriptorRange> descriptor_ranges;

// LOCAL (video): X2W_NOOP_CENSUS=1 counts the calls of every unimplemented (no-op) slot per object
// kind, and logs the first calls whose register or stack arguments (or the first 64 bytes they point
// to, for the first 64 calls of each slot) hold a descriptor handle inside a title or desktop heap
// range. A console "create view" or "write descriptors" entry point would show up here with handles
// into the main heap. At most 6 hits per slot and 240 in all; stack words are marked "weak" (stale
// caller stack often holds old pointers). Totals at 1024, 2048, 4096, ... no-op calls, one line per kind.
// The same switch logs views that view_slot drops because their destination does not translate.
LONG64 noop_census[16][512];
LONG noop_census_hits[16][512];
LONG64 noop_census_total;
bool noop_census_on()
{
    static const bool on = [] {
        char value[4]{};
        return GetEnvironmentVariableA("X2W_NOOP_CENSUS", value, sizeof(value)) && value[0] == '1';
    }();
    return on;
}
void log(const char* format, ...);
bool first_calls(LONG& counter, LONG limit);
int range_watch_count();                                    // X2W_RANGE_WATCH, below
int range_watch_hit(std::uint64_t address, std::uint64_t bytes);
bool range_watch_line(int i);
std::uint64_t range_watch_address(int i);
extern LONG frame_number;
bool noop_census_enabled()
{
    static const bool on = [] {
        if (noop_census_on()) log("x2w-noop-census on: counting no-op slot calls and descriptor handles in their arguments");
        return noop_census_on();
    }();
    return on;
}
struct CensusHeap {
    std::uint64_t base, index, count;
    bool desktop;
};
bool noop_census_heap_index(std::uint64_t value, CensusHeap* out)
{
    if (value < 0x10000) return false;
    bool found = false;
    AcquireSRWLockShared(&descriptor_lock);
    for (const auto& r : descriptor_ranges) {
        if (r.title >= 0x10000 && r.title_stride && value >= r.title && value < r.title + r.bytes) {
            *out = {r.title, (value - r.title) / r.title_stride, r.bytes / r.title_stride, false};
            found = true;
            break;
        }
        if (r.desktop >= 0x10000 && r.desktop_stride && value >= r.desktop && value < r.desktop + r.desktop_descriptors * r.desktop_stride) {
            *out = {r.desktop, (value - r.desktop) / r.desktop_stride, r.desktop_descriptors, true};
            found = true;
            break;
        }
    }
    ReleaseSRWLockShared(&descriptor_lock);
    return found;
}
bool noop_census_hit(int kind, int slot)
{
    static LONG hits;
    return hits < 240 && InterlockedIncrement(&noop_census_hits[kind & 15][slot & 511]) <= 6 && first_calls(hits, 240);
}
void noop_census_note(int kind, int slot, const std::uint64_t* args, std::uint64_t ret)
{
    const LONG64 calls = InterlockedIncrement64(&noop_census[kind & 15][slot & 511]);
    const LONG64 total = InterlockedIncrement64(&noop_census_total);
    static const char* const names[8] = {"rcx", "rdx", "r8", "r9", "s0", "s1", "s2", "s3"};
    for (int i = 1; i < 8 && range_watch_count(); ++i) {
        const int r = range_watch_hit(args[i], 1);
        if (r >= 0 && range_watch_line(r))
            log("x2w-watch 0x%llx: no-op %s::slot%d call %lld %s=0x%llx frame %ld; ret=+0x%llx", range_watch_address(r), kind_names[kind],
                slot, calls, names[i], args[i], frame_number, ret);
        std::uint64_t words[8]{};
        SIZE_T got = 0;
        if (args[i] < 0x10000 || r >= 0 ||
            !ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(args[i]), words, sizeof(words), &got) || got != sizeof(words))
            continue;
        for (int w = 0; w < 8; ++w) {
            const int rw = range_watch_hit(words[w], 1);
            if (rw >= 0 && range_watch_line(rw))
                log("x2w-watch 0x%llx: no-op %s::slot%d call %lld [%s+0x%x]=0x%llx frame %ld; ret=+0x%llx", range_watch_address(rw),
                    kind_names[kind], slot, calls, names[i], w * 8, words[w], frame_number, ret);
        }
    }
    for (int i = 1; i < 8 && noop_census_hits[kind & 15][slot & 511] < 6; ++i) {
        CensusHeap h{};
        const char* weak = i >= 4 ? " (stack word: weak)" : "";
        if (noop_census_heap_index(args[i], &h) && noop_census_hit(kind, slot))
            log("x2w-noop-census %s::slot%d call %lld: %s=0x%llx is %s index 0x%llx of the heap at 0x%llx (%llu descriptors)%s; ret=+0x%llx",
                kind_names[kind], slot, calls, names[i], args[i], h.desktop ? "DESKTOP" : "title", h.index, h.base, h.count, weak, ret);
        if (calls > 64 || args[i] < 0x10000) continue;
        std::uint64_t words[8]{};
        SIZE_T got = 0;
        if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(args[i]), words, sizeof(words), &got) || got != sizeof(words)) continue;
        for (int w = 0; w < 8; ++w)
            if (noop_census_heap_index(words[w], &h) && noop_census_hit(kind, slot))
                log("x2w-noop-census %s::slot%d call %lld: [%s+0x%x]=0x%llx is %s index 0x%llx of the heap at 0x%llx (%llu descriptors)%s; ret=+0x%llx",
                    kind_names[kind], slot, calls, names[i], w * 8, words[w], h.desktop ? "DESKTOP" : "title", h.index, h.base, h.count, weak, ret);
    }
    if (total >= 1024 && (total & (total - 1)) == 0) {
        for (int k = 0; k < kind_count; ++k) {
            char text[1536];
            int at = std::snprintf(text, sizeof(text), "x2w-noop-census after %lld no-op calls, %s:", total, kind_names[k]);
            bool any = false;
            for (int s = 0; s < 512 && at < static_cast<int>(sizeof(text)) - 32; ++s)
                if (noop_census[k][s]) {
                    at += std::snprintf(text + at, sizeof(text) - at, " %d=%lld", s, noop_census[k][s]);
                    any = true;
                }
            if (any) log("%s", text);
        }
    }
}
// Views view_slot drops because the destination is in no known heap (desktop_cpu_handle gives 0).
void noop_census_dropped_view(int slot, std::uint64_t title_dest)
{
    static LONG64 dropped;
    static LONG logged;
    const LONG64 n = InterlockedIncrement64(&dropped);
    if (first_calls(logged, 32) || (n & (n - 1)) == 0)
        log("x2w-noop-census view slot %d dropped (destination 0x%llx is in no known heap); %lld dropped so far", slot, title_dest, n);
}

// LOCAL (video): X2W_RANGE_WATCH=<hex title address>[,<hex>...] (up to 4) watches [address, address + 0x100):
// - its bytes at slot-133 creation of a placed buffer that overlaps it;
// - its bytes at every queue submit, frame and draw/compute bind, logged only when they changed since the last
//   sample (site, frame, first 48 bytes, hash of all 0x100);
// - every translated compute binding and every no-op census argument (or one word it points to) in the range.
// At most 64 lines per watched address.
extern LONG frame_number;
constexpr std::uint64_t range_watch_span = 0x100;
struct RangeWatch {
    std::uint64_t address;
    std::uint64_t hash;
    bool sampled;
    LONG lines;
};
RangeWatch range_watches[4];
int range_watch_count()
{
    static const int count = [] {
        char text[128]{};
        int n = 0;
        if (GetEnvironmentVariableA("X2W_RANGE_WATCH", text, sizeof(text))) {
            for (char* at = text; *at && n < 4;) {
                char* end = nullptr;
                const std::uint64_t value = std::strtoull(at, &end, 16);
                if (end == at) break;
                if (value) range_watches[n++].address = value;
                at = *end == ',' ? end + 1 : end;
            }
            log("x2w-watch X2W_RANGE_WATCH: %d range(s) of 0x%llx bytes watched (%s)", n, range_watch_span, text);
        }
        return n;
    }();
    return count;
}
// Index of the first watched range that [address, address + bytes) overlaps, else -1.
int range_watch_hit(std::uint64_t address, std::uint64_t bytes)
{
    if (!range_watch_count() || !address) return -1;
    if (!bytes) bytes = 1;
    for (int i = 0; i < range_watch_count(); ++i) {
        const std::uint64_t a = range_watches[i].address;
        if (address < a + range_watch_span && address + bytes > a) return i;
    }
    return -1;
}
bool range_watch_line(int i) { return InterlockedIncrement(&range_watches[i].lines) <= 64; }
std::uint64_t range_watch_address(int i) { return range_watches[i].address; }
// Reads the watched range without faulting; false if it is not readable.
bool range_watch_read(int i, unsigned char (&bytes)[range_watch_span], std::uint64_t* hash)
{
    SIZE_T got = 0;
    if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(range_watches[i].address), bytes, sizeof(bytes), &got) ||
        got != sizeof(bytes))
        return false;
    std::uint64_t h = 14695981039346656037ull;
    for (unsigned char b : bytes) h = (h ^ b) * 1099511628211ull;
    *hash = h;
    return true;
}
void range_watch_text(const unsigned char* bytes, char (&text)[3 * 48 + 1])
{
    for (int k = 0; k < 48; ++k) std::snprintf(text + 3 * k, 4, "%02x ", bytes[k]);
}
// Samples every watched range; logs the ones whose bytes changed since the last sample (or the first sample).
void range_watch_sample(const char* site, std::uint64_t key = 0)
{
    for (int i = 0; i < range_watch_count(); ++i) {
        unsigned char bytes[range_watch_span];
        std::uint64_t hash = 0;
        if (!range_watch_read(i, bytes, &hash)) continue;
        RangeWatch& w = range_watches[i];
        if (w.sampled && w.hash == hash) continue;
        const bool first = !w.sampled;
        w.sampled = true;
        w.hash = hash;
        if (!range_watch_line(i)) continue;
        char text[3 * 48 + 1];
        range_watch_text(bytes, text);
        log("x2w-watch 0x%llx %s at %s %016llx frame %ld: hash %016llx [%s]", w.address, first ? "first sample" : "CHANGED", site, key,
            frame_number, hash, text);
    }
}

// Desktop rejects a shader-visible CBV/SRV/UAV heap above a driver-specific descriptor count: the
// API documents 1,048,576, but a driver may accept fewer (1,000,000 on the GTX 1660 SUPER here), so
// the accepted maximum is measured once and cached rather than assumed.
constexpr std::uint64_t view_heap_probe_floor = 65536;

void log(const char* format, ...);
bool first_calls(LONG& counter, LONG limit);
void request_translation(Wrapper* pipeline);
struct RootParameter;
void request_slot57_mesh(Wrapper* pipeline, const std::vector<RootParameter>* bound_root);
std::uint64_t accepted_view_heap_descriptors(ID3D12Device* device, UINT node_mask);

std::uint64_t desktop_cpu_handle(std::uint64_t handle, UINT* contiguous = nullptr, UINT* available = nullptr)
{
    std::uint64_t result = handle;
    UINT room = UINT_MAX, remaining = UINT_MAX;
    bool past_end = false;
    AcquireSRWLockShared(&descriptor_lock);
    for (const auto& r : descriptor_ranges) {
        if (handle < r.title) continue;
        const u64 offset = handle - r.title;
        if (offset == r.bytes && r.tail_cpu) past_end = true;
        if (offset >= r.bytes) continue;
        const u64 index = offset / r.title_stride;
        const u64 limit = r.bytes / r.title_stride;
        remaining = static_cast<UINT>(limit - index);
        if (r.tail_cpu && offset % r.title_stride) {
            result = 0;
            room = 0;
        } else if (index >= r.desktop_descriptors) {
            result = r.tail_cpu + (index - r.desktop_descriptors) * r.desktop_stride;
            room = static_cast<UINT>(limit - index);
        } else {
            result = r.desktop + index * r.desktop_stride + offset % r.title_stride;
            room = static_cast<UINT>(r.desktop_descriptors - index);
        }
        past_end = false;
        break;
    }
    ReleaseSRWLockShared(&descriptor_lock);
    if (past_end) { result = 0; room = 0; }
    if (contiguous) *contiguous = room;
    if (available) *available = result ? remaining : 0;
    return result;
}

// Measure, once per process, the largest shader-visible CBV/SRV/UAV heap this driver creates. The
// documented ceiling is 1,048,576 descriptors, but drivers may accept fewer, so halve from the
// ceiling until a create succeeds and then bisect upward. Each probe heap is released before the next
// one, so at most one probe holds memory at a time.
std::uint64_t accepted_view_heap_descriptors(ID3D12Device* device, UINT node_mask)
{
    static std::uint64_t cached;
    static LONG state;  // 0 none, 1 measuring, 2 measured
    if (InterlockedCompareExchange(&state, 0, 0) == 2) return cached;
    if (InterlockedCompareExchange(&state, 1, 0) != 0) {
        // Another thread is measuring; wait for it rather than create a second set of probe heaps.
        for (int spin = 0; spin < 2000 && InterlockedCompareExchange(&state, 0, 0) != 2; ++spin) Sleep(1);
        return InterlockedCompareExchange(&state, 0, 0) == 2 ? cached : 0;
    }

    const auto probe = [device, node_mask](std::uint64_t count) {
        D3D12_DESCRIPTOR_HEAP_DESC desc{};
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        desc.NumDescriptors = static_cast<UINT>(count);
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        desc.NodeMask = node_mask;
        ID3D12DescriptorHeap* heap = nullptr;
        const bool ok = SUCCEEDED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap))) && heap;
        if (heap) heap->Release();
        return ok;
    };

    // Halve from the documented ceiling, testing the floor too: a driver whose maximum is below the
    // floor must not be recorded as accepting it, or the retry would ask for a size it rejects again.
    std::uint64_t accepted = 0;
    for (std::uint64_t size = 1048576; size >= view_heap_probe_floor; size /= 2) {
        if (probe(size)) {
            accepted = size;
            break;
        }
    }
    if (!accepted) {
        InterlockedExchange(&state, 2);
        log("x2w-d3d12 no probe size at or above %llu was accepted; the view heap shortcut is declined",
            view_heap_probe_floor);
        return 0;
    }
    std::uint64_t low = accepted, high = accepted * 2;
    while (low + 1 < high && high <= 1048576) {
        const std::uint64_t mid = low + (high - low) / 2;
        if (probe(mid)) low = mid; else high = mid;
    }
    cached = low;
    InterlockedExchange(&state, 2);
    log("x2w-d3d12 measured the shader-visible view heap limit: %llu descriptors", cached);
    return cached;
}

UINT title_descriptor_stride(u64 handle, UINT desktop_stride)
{
    AcquireSRWLockShared(&descriptor_lock);
    UINT result = desktop_stride;
    for (const auto& range : descriptor_ranges)
        if (handle >= range.title && handle - range.title < range.bytes) result = range.title_stride;
    ReleaseSRWLockShared(&descriptor_lock);
    return result;
}

void* tables[kind_count][slot_count];
Wrapper* the_device;

// Time spent inside the stand-in per entry-point family, reported with the heartbeat.
enum { time_list, time_device_noop, time_forward, time_create, time_view, time_queue, time_present, time_wait,
       time_pipeline, time_heap, time_copy, time_cx_fill_views, time_cx_desktop_range, time_cx_readable, time_cx_texture,
       time_cx_execute, time_vquery, time_placed, time_placed_query, time_placed_create, time_placed_log, time_placed_gap,
       time_draw_lock_wait, time_log, time_log_in_draw_lock, time_count };
// LOCAL (perf_work/STARTUP.md): "placed" is all of Device slot 133 (CreatePlacedResourceX); its parts are
// the memory queries, the desktop heap/resource creation and the log line; "placed-gap" is the time on
// the same thread from the end of one slot 133 call to the start of the next (the title's own work
// between two calls; gaps over 1 s are not counted).
const char* const time_names[time_count] = {"list", "device-noop", "forward", "create", "view", "queue", "present", "wait",
                                            "pipeline", "heap", "copy", "cx-fill-views", "cx-desktop-range", "cx-readable",
                                            "cx-texture-view", "cx-execute", "vquery", "placed", "placed-query",
                                            "placed-create", "placed-log", "placed-gap", "draw-lock-wait", "log",
                                            "log-in-draw-lock"};
volatile LONG64 time_ticks[time_count], time_calls[time_count];
struct Timed;
SIZE_T timed_query(const void* address, MEMORY_BASIC_INFORMATION* info);  // VirtualQuery, timed as "vquery"
volatile LONG64 list_slot_ticks[256], list_slot_calls[256];  // per command-list slot
volatile LONG64 list_slot_census[256];  // per command-list slot, reported with the render totals
// LOCAL stall: per-thread time inside timed stand-in calls (the outermost Timed scope only, so nested
// categories are not counted twice), for the present split line (present_split).
struct ThreadInside {
    volatile LONG tid;
    volatile LONG64 ticks;
};
constexpr LONG thread_inside_max = 256;
ThreadInside thread_inside[thread_inside_max];
volatile LONG thread_inside_count;
thread_local ThreadInside* thread_inside_mine;
thread_local int timed_depth;
ThreadInside* inside_slot()
{
    if (!thread_inside_mine) {
        static ThreadInside overflow;
        const LONG i = InterlockedIncrement(&thread_inside_count) - 1;
        thread_inside_mine = i < thread_inside_max ? &thread_inside[i] : &overflow;
        thread_inside_mine->tid = static_cast<LONG>(GetCurrentThreadId());
    }
    return thread_inside_mine;
}
struct Timed {
    int category;
    LARGE_INTEGER start;
    explicit Timed(int c) : category(c)
    {
        ++timed_depth;
        QueryPerformanceCounter(&start);
    }
    ~Timed()
    {
        LARGE_INTEGER end;
        QueryPerformanceCounter(&end);
        InterlockedAdd64(&time_ticks[category], end.QuadPart - start.QuadPart);
        InterlockedIncrement64(&time_calls[category]);
        if (--timed_depth == 0) InterlockedAdd64(&inside_slot()->ticks, end.QuadPart - start.QuadPart);
    }
};

// LOCAL: a part of a call timed into one category (Timed covers a whole scope).
struct TimedPart {
    int category;
    LARGE_INTEGER start;
    explicit TimedPart(int c) : category(c) { QueryPerformanceCounter(&start); }
    void stop()
    {
        if (category < 0) return;
        LARGE_INTEGER end;
        QueryPerformanceCounter(&end);
        InterlockedAdd64(&time_ticks[category], end.QuadPart - start.QuadPart);
        InterlockedIncrement64(&time_calls[category]);
        category = -1;
    }
    ~TimedPart() { stop(); }
};
// Slot 133 totals since the start, never reset (the per-report times above are).
volatile LONG64 placed_calls_total, placed_ticks_total;

SIZE_T timed_query(const void* address, MEMORY_BASIC_INFORMATION* info)
{
    Timed timed(time_vquery);
    return VirtualQuery(address, info, sizeof(*info));
}

extern SRWLOCK draw_lock;              // d3d12_draw.inl
extern volatile LONG draw_lock_thread;  // the thread that took draw_lock last
// LOCAL stall: all of log() is timed as "log"; lines written while this thread holds draw_lock also
// as "log-in-draw-lock" (owner check: the last taker is this thread and the lock is not free).
void log(const char* format, ...)
{
    Timed timed(time_log);
    TimedPart locked(time_log_in_draw_lock);
    if (draw_lock_thread != static_cast<LONG>(GetCurrentThreadId()))
        locked.category = -1;
    else if (TryAcquireSRWLockShared(&draw_lock)) {
        ReleaseSRWLockShared(&draw_lock);
        locked.category = -1;
    }
    char line[1536];
    va_list args;
    va_start(args, format);
    int n = std::vsnprintf(line, sizeof(line) - 2, format, args);
    va_end(args);
    if (n < 0) return;
    if (n > static_cast<int>(sizeof(line)) - 3) n = sizeof(line) - 3;
    line[n++] = '\r';
    line[n++] = '\n';
    x2w_trap::write_all(GetStdHandle(STD_ERROR_HANDLE), line, static_cast<DWORD>(n));
    x2w_trap::append_trap_log(line, static_cast<DWORD>(n));  // one open handle per module (trap.h)
}

void guid_text(const GUID* g, char* out, std::size_t size)
{
    if (!g || !x2w_local::readable(g, sizeof(GUID))) {
        std::snprintf(out, size, "%p", static_cast<const void*>(g));
        return;
    }
    std::snprintf(out, size, "%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x", g->Data1, g->Data2, g->Data3,
                  g->Data4[0], g->Data4[1], g->Data4[2], g->Data4[3], g->Data4[4], g->Data4[5], g->Data4[6],
                  g->Data4[7]);
}

void hex_text(const void* data, std::size_t bytes, char* out, std::size_t size)
{
    out[0] = 0;
    if (!data || !x2w_local::readable(data, bytes)) return;
    const auto* p = static_cast<const unsigned char*>(data);
    std::size_t used = 0;
    for (std::size_t i = 0; i < bytes && used + 4 < size; ++i) used += std::snprintf(out + used, size - used, "%02x ", p[i]);
}

bool first_calls(LONG& counter, LONG limit = 3) { return x2w_local::first_calls(counter, limit); }

// X2W_D3D12_DEBUG=1: the desktop debug layer's messages are drained into the log after forwarded
// calls, tagged with the call that preceded them, and a device removal is reported once with its reason.
ID3D12InfoQueue* info_queue;
ID3D12Device* removal_device;
LONG removal_reported, debug_lines;
SRWLOCK debug_lock = SRWLOCK_INIT;

void report_dred_dispatches(const D3D12_AUTO_BREADCRUMB_NODE* node, UINT done);  // LOCAL, by execute_dispatch
// LOCAL: logs DRED breadcrumbs (the operations around the last completed one, per list) and the page fault.
void report_dred(ID3D12Device* device)
{
    ID3D12DeviceRemovedExtendedData* dred = nullptr;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&dred)))) {
        log("x2w-dred not available");
        return;
    }
    D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT crumbs{};
    if (SUCCEEDED(dred->GetAutoBreadcrumbsOutput(&crumbs))) {
        int lists = 0;
        for (const D3D12_AUTO_BREADCRUMB_NODE* node = crumbs.pHeadAutoBreadcrumbNode; node && lists < 24; node = node->pNext) {
            const UINT done = node->pLastBreadcrumbValue ? *node->pLastBreadcrumbValue : 0;
            if (done >= node->BreadcrumbCount) continue;  // completed list
            ++lists;
            log("x2w-dred list %p: %u of %u operations completed", static_cast<const void*>(node->pCommandList), done, node->BreadcrumbCount);
            const UINT from = done > 6 ? done - 6 : 0, to = done + 4 < node->BreadcrumbCount ? done + 4 : node->BreadcrumbCount;
            for (UINT i = from; i < to; ++i)
                log("x2w-dred   %s op %u: %d", i == done ? ">>" : "  ", i, static_cast<int>(node->pCommandHistory[i]));
            report_dred_dispatches(node, done);
        }
        if (!lists) log("x2w-dred no incomplete list in the breadcrumbs");
    }
    D3D12_DRED_PAGE_FAULT_OUTPUT fault{};
    if (SUCCEEDED(dred->GetPageFaultAllocationOutput(&fault))) {
        log("x2w-dred page fault va 0x%llx", static_cast<unsigned long long>(fault.PageFaultVA));
        int n = 0;
        for (const D3D12_DRED_ALLOCATION_NODE* a = fault.pHeadExistingAllocationNode; a && n < 8; a = a->pNext, ++n)
            log("x2w-dred   existing allocation type %d", static_cast<int>(a->AllocationType));
        n = 0;
        for (const D3D12_DRED_ALLOCATION_NODE* a = fault.pHeadRecentFreedAllocationNode; a && n < 8; a = a->pNext, ++n)
            log("x2w-dred   recently freed allocation type %d", static_cast<int>(a->AllocationType));
    }
    dred->Release();
}

void check_device(const char* context, int slot)
{
    if (removal_device && !removal_reported) {
        const HRESULT reason = removal_device->GetDeviceRemovedReason();
        if (reason != S_OK && InterlockedExchange(&removal_reported, 1) == 0) {
            log("x2w-d3d12 DEVICE REMOVED, reason 0x%08lx, first noticed after %s %d", static_cast<unsigned long>(reason),
                context, slot);
            report_dred(removal_device);
        }
    }
    if (!info_queue) return;
    AcquireSRWLockExclusive(&debug_lock);
    const UINT64 count = info_queue->GetNumStoredMessages();
    static std::unordered_map<int, int> per_id;  // LOCAL: at most 4 lines per message id
    for (UINT64 i = 0; i < count && debug_lines < 4000; ++i) {
        SIZE_T size = 0;
        info_queue->GetMessage(i, nullptr, &size);
        if (!size || size > 8192) continue;
        alignas(16) char storage[8192];
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage);
        if (SUCCEEDED(info_queue->GetMessage(i, message, &size)) && ++per_id[message->ID] <= 4) {
            ++debug_lines;
            log("x2w-d3d12-debug [%s %d] sev=%d id=%d %.900s", context, slot, message->Severity, message->ID,
                message->pDescription);
        }
    }
    info_queue->ClearStoredMessages();
    ReleaseSRWLockExclusive(&debug_lock);
}

// Wrapper detection is a registry lookup: no argument is ever dereferenced to find out what it is.
SRWLOCK registry_lock = SRWLOCK_INIT;
std::unordered_set<const void*>& registry()
{
    static std::unordered_set<const void*> wrappers;
    return wrappers;
}

Wrapper* as_wrapper(const void* p)
{
    const auto value = reinterpret_cast<std::uintptr_t>(p);
    if (value < 0x10000 || (value & 7)) return nullptr;
    AcquireSRWLockShared(&registry_lock);
    const bool known = registry().count(p) != 0;
    ReleaseSRWLockShared(&registry_lock);
    return known ? const_cast<Wrapper*>(static_cast<const Wrapper*>(p)) : nullptr;
}

#include "startup_stubs/d3d12_cold_texture.inl"

u64 unwrap(u64 value)
{
    Wrapper* w = as_wrapper(reinterpret_cast<const void*>(value));
    warm_texture(w);
    return w ? reinterpret_cast<u64>(w->own().real) : value;
}

Wrapper* wrap(IUnknown* real, Kind kind)
{
    const std::size_t area = kind == list_kind ? 0x10000 : sizeof(Wrapper);
    auto* block = static_cast<unsigned char*>(_aligned_malloc(sizeof(WrapperOwn) + area, 16));
    std::memset(block, 0, sizeof(WrapperOwn) + area);
    auto* w = reinterpret_cast<Wrapper*>(block + sizeof(WrapperOwn));
    w->vtable = tables[kind];
    w->own().real = real;
    w->own().refs = 1;
    w->own().kind = kind;
    if (kind == fence_kind && real) w->own().completed = static_cast<LONG64>(static_cast<ID3D12Fence*>(real)->GetCompletedValue());
    AcquireSRWLockExclusive(&registry_lock);
    registry().insert(w);
    ReleaseSRWLockExclusive(&registry_lock);
    return w;
}

Kind kind_for(REFIID iid)
{
    if (iid == __uuidof(ID3D12Device) || iid == __uuidof(ID3D12Device1) || iid == __uuidof(ID3D12Device2) ||
        iid == __uuidof(ID3D12Device3) || iid == __uuidof(ID3D12Device4) || iid == __uuidof(ID3D12Device5) ||
        iid == __uuidof(ID3D12Device6) || iid == __uuidof(ID3D12Device7) || iid == __uuidof(ID3D12Device8) ||
        iid == __uuidof(ID3D12Device9) || iid == __uuidof(ID3D12Device10))
        return device_kind;
    // ID3D12Device11 arrived after the 10.0.22621 SDK headers; match its IID literally.
    static const GUID device11 = {0x5405c344, 0xd457, 0x444e, {0xb4, 0xdd, 0x23, 0x66, 0xe4, 0x5a, 0xee, 0x39}};
    if (iid == device11) return device_kind;
    if (iid == __uuidof(ID3D12CommandQueue)) return queue_kind;
    if (iid == __uuidof(ID3D12Fence) || iid == __uuidof(ID3D12Fence1)) return fence_kind;
    if (iid == __uuidof(ID3D12CommandAllocator)) return allocator_kind;
    if (iid == __uuidof(ID3D12CommandList) || iid == __uuidof(ID3D12GraphicsCommandList) ||
        iid == __uuidof(ID3D12GraphicsCommandList1) || iid == __uuidof(ID3D12GraphicsCommandList2) ||
        iid == __uuidof(ID3D12GraphicsCommandList3) || iid == __uuidof(ID3D12GraphicsCommandList4) ||
        iid == __uuidof(ID3D12GraphicsCommandList5) || iid == __uuidof(ID3D12GraphicsCommandList6) ||
        iid == __uuidof(ID3D12GraphicsCommandList7))
        return list_kind;
    // Xbox-only command list interface requested from device slot 73 (not in the desktop SDK header).
    static const GUID xbox_list = {0xc3827890, 0x4024, 0x474c, {0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74}};
    if (iid == xbox_list) return list_kind;
    if (iid == __uuidof(ID3D12DescriptorHeap)) return descriptor_heap_kind;
    if (iid == __uuidof(ID3D12RootSignature)) return root_signature_kind;
    if (iid == __uuidof(ID3D12Resource) || iid == __uuidof(ID3D12Resource1) || iid == __uuidof(ID3D12Resource2))
        return resource_kind;
    if (iid == __uuidof(ID3D12Heap) || iid == __uuidof(ID3D12Heap1)) return heap_kind;
    if (iid == __uuidof(ID3D12QueryHeap)) return query_heap_kind;
    if (iid == __uuidof(ID3D12PipelineState)) return pipeline_kind;
    if (iid == __uuidof(ID3D12CommandSignature)) return command_signature_kind;
    return other_kind;
}

bool is_base_iid(REFIID iid)
{
    return iid == __uuidof(IUnknown) || iid == __uuidof(ID3D12Object) || iid == __uuidof(ID3D12DeviceChild) ||
           iid == __uuidof(ID3D12Pageable);
}

// Traps. The kind is a template argument so a trap never dereferences its arguments.
template <int K, int N>
__declspec(noinline) u64 trap_slot(u64 rcx, u64 rdx, u64 r8, u64 r9)
{
    char name[64];
    std::snprintf(name, sizeof(name), "%s::slot%d", kind_names[K], N);
    x2w_trap::fire(X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, name, (0x80 | N) & 0xFF, rcx, rdx, r8, r9, _ReturnAddress());
}

template <int K, std::size_t... I>
void fill_traps(void** table, std::index_sequence<I...>)
{
    ((table[I] = reinterpret_cast<void*>(&trap_slot<K, static_cast<int>(I)>)), ...);
}

template <std::size_t... K>
void fill_all_traps(std::index_sequence<K...>)
{
    (fill_traps<static_cast<int>(K)>(tables[K], std::make_index_sequence<slot_count>{}), ...);
}

// A call that passes a known interface IID is a creation or query call: a no-op that returned 0
// would hand the title "success" with no object. Such calls trap instead of being ignored.
bool names_known_iid(u64 value)
{
    if (value < 0x10000 || !x2w_local::readable(reinterpret_cast<const void*>(value), sizeof(GUID))) return false;
    return kind_for(*reinterpret_cast<const GUID*>(value)) != other_kind;
}
bool create_shaped(u64 a, u64 b, u64 c, u64 d, u64 e, u64 f, u64 g)
{
    return names_known_iid(a) || names_known_iid(b) || names_known_iid(c) || names_known_iid(d) ||
           names_known_iid(e) || names_known_iid(f) || names_known_iid(g);
}

bool list_noop()
{
    char value[4]{};
    return GetEnvironmentVariableA("X2W_D3D12_LIST_NOOP", value, sizeof(value)) && value[0] == '1';
}

// With X2W_D3D12_LIST_NOOP=1: an unknown command-list method records nothing and returns 0.
extern LONG draw_calls_seen;
// X2W_D3D12_TRACE_FRAME=<n>: every command-list call of presented frame n is logged ("x2w-frame")
// with its slot, the kinds of any wrapped objects among its arguments, and stack arguments.
// LOCAL (scene): X2W_D3D12_TRACE_ALL=1 traces from the start (fixtures that never present).
LONG trace_requested_frame()
{
    static const LONG wanted = [] {
        char value[16]{};
        return GetEnvironmentVariableA("X2W_D3D12_TRACE_FRAME", value, sizeof(value)) ? std::atol(value) : 0L;
    }();
    return wanted;
}
LONG trace_frame_active = x2w_local::env_on("X2W_D3D12_TRACE_ALL") || trace_requested_frame() == 1 ? 1 : 0;
volatile LONG gpu_trace_frame = trace_requested_frame() == 1 ? 1 : 0;
volatile LONG gpu_trace_commands;
volatile LONG gpu_trace_per_kind[2];  // graphics and compute each have a trace budget
volatile LONG gpu_trace_bindings;
volatile LONG gpu_trace_request_id;
using TraceStageKey = std::shared_ptr<const std::string>;
TraceStageKey gpu_scene_stage_override;
TraceStageKey gpu_scene_second_override;
TraceStageKey gpu_trace_stage_key()
{
    if (auto active = std::atomic_load(&gpu_scene_stage_override)) return active;
    static const TraceStageKey key = []() -> TraceStageKey {
        char value[32]{};
        const DWORD n = GetEnvironmentVariableA("X2W_TRACE_STAGE_KEY", value, sizeof(value));
        if (!n) return {};
        if (n == 16 && std::strspn(value, "0123456789abcdefABCDEF") == 16) return std::make_shared<const std::string>(value);
        log("x2w-frame X2W_TRACE_STAGE_KEY requires a 16-digit hexadecimal shader key");
        return std::make_shared<const std::string>("invalid");  // An invalid filter selects no shaders.
    }();
    return key;
}
void reset_gpu_trace_commands()
{
    InterlockedExchange(&gpu_trace_commands, 0);
    InterlockedExchange(&gpu_trace_bindings, 0);
    for (auto& count : gpu_trace_per_kind) InterlockedExchange(&count, 0);
}

// A scene trace begins on an issued deferred draw. Presentation polls the
// optional request file and closes each capture on its next Present.
struct GpuSceneTrace {
    std::mutex lock;
    std::atomic<bool> waiting{false};
    LONG mrt = 0, after = 0, first = 0, accepted = 0, pending = 0, active_frame = 0, active_id = 0;
    std::atomic<ULONGLONG> next_poll{0};
    std::wstring request_file;
    TraceStageKey pending_key;
    TraceStageKey second_key;
    x2w_gpu::CapturePoint pending_point;

    GpuSceneTrace()
    {
        char text[32]{};
        const auto bytes = GetEnvironmentVariableA("X2W_TRACE_SCENE_MRT", text, sizeof(text));
        if (bytes && bytes < sizeof(text)) mrt = std::clamp(std::atol(text), 0L, 8L);
        const auto delay = GetEnvironmentVariableA("X2W_D3D12_TRACE_BUSY_AFTER", text, sizeof(text));
        if (delay && delay < sizeof(text)) after = (std::max)(0L, std::atol(text));
        wchar_t name[MAX_PATH]{};
        const auto length = GetEnvironmentVariableW(L"X2W_TRACE_REQUEST_FILE", name, MAX_PATH);
        if (length && length < MAX_PATH) request_file.assign(name, length);
        waiting.store(mrt != 0);
    }

    void poll()
    {
        if (request_file.empty()) return;
        const auto now = GetTickCount64();
        auto due = next_poll.load(std::memory_order_relaxed);
        if (now < due || !next_poll.compare_exchange_strong(due, now + 250, std::memory_order_relaxed)) return;
        FILE* f = nullptr;
        if (_wfopen_s(&f, request_file.c_str(), L"rb") != 0) return;
        char text[64]{};
        const auto n = std::fread(text, 1, sizeof(text) - 1, f);
        std::fclose(f);
        if (!n || n == sizeof(text) - 1) return;
        const auto request = x2w_gpu::parse_scene_request({text, n});
        if (!request) return;
        const auto id = request->id;
        std::lock_guard<std::mutex> held(lock);
        if (id <= accepted || pending || active_frame) return;
        pending_point = request->point;
        pending_key = request->stage_key.empty() ? nullptr : std::make_shared<const std::string>(request->stage_key);
        second_key = request->second_key.empty() ? nullptr : std::make_shared<const std::string>(request->second_key);
        accepted = pending = id;
        waiting.store(true, std::memory_order_release);
        log("x2w-frame scene request %ld armed for the next qualifying draw, stage=%s%s%s", id,
            pending_key ? pending_key->c_str() : "default", second_key ? "," : "", second_key ? second_key->c_str() : "");
    }
};
GpuSceneTrace& gpu_scene_trace() { static GpuSceneTrace trace; return trace; }
extern LONG capture_active;
extern LONG frame_number;
// Command-list methods with a real implementation (render mode). Every list call enters through
// noop_slot, which times, captures and traces it, then hands over to the entry here if there is one.
using ListTyped = u64(__stdcall*)(u64, u64, u64, u64, u64, u64, u64, u64);
ListTyped list_typed[slot_count];
bool compute_binding_trace_enabled();
void trace_compute_binding(Wrapper*, int slot, u64 first, u64 second, u64 third, u64 fourth);
namespace capture { void list_call(int slot, u64 list, u64 rdx, u64 r8, u64 r9, const u64* stack, u64 ret); }
const char* arg_kind(u64 value)
{
    Wrapper* w = as_wrapper(reinterpret_cast<const void*>(value));
    return w ? kind_names[w->own().kind] : "";
}
template <int K, int N>
__declspec(noinline) u64 noop_slot(u64 rcx, u64 rdx, u64 r8, u64 r9, u64 s0, u64 s1, u64 s2, u64 s3)
{
    Timed timed(K == list_kind ? time_list : K == queue_kind ? time_queue : time_device_noop);
    struct SlotTimed {
        LARGE_INTEGER start;
        SlotTimed() { QueryPerformanceCounter(&start); }
        ~SlotTimed()
        {
            LARGE_INTEGER end;
            QueryPerformanceCounter(&end);
            if (K == list_kind) {
                InterlockedAdd64(&list_slot_ticks[N & 255], end.QuadPart - start.QuadPart);
                InterlockedIncrement64(&list_slot_calls[N & 255]);
                InterlockedIncrement64(&list_slot_census[N & 255]);
            }
        }
    } slot_timed;
    if (K == list_kind) InterlockedIncrement(&draw_calls_seen);
    if (K == list_kind && capture_active) {
        const u64 stack[4] = {s0, s1, s2, s3};
        capture::list_call(N, rcx, rdx, r8, r9, stack,
                           reinterpret_cast<u64>(_ReturnAddress()) - reinterpret_cast<u64>(GetModuleHandleW(nullptr)));
    }
    if (K == list_kind && trace_frame_active) {
        const auto ret = reinterpret_cast<u64>(_ReturnAddress()) - reinterpret_cast<u64>(GetModuleHandleW(nullptr));
        log("x2w-frame tid=%lu list=%llx slot%d(0x%llx%s%s, 0x%llx%s%s, 0x%llx%s%s | 0x%llx, 0x%llx, 0x%llx, 0x%llx) ret=+0x%llx",
            GetCurrentThreadId(), rcx, N, rdx, *arg_kind(rdx) ? ":" : "", arg_kind(rdx), r8, *arg_kind(r8) ? ":" : "",
            arg_kind(r8), r9, *arg_kind(r9) ? ":" : "", arg_kind(r9), s0, s1, s2, s3, ret);
    }
    if constexpr (K == list_kind && (N == 14 || N == 35 || N == 37 || N == 39 || N == 41 || N == 43 || N == 45)) {
        if (compute_binding_trace_enabled()) trace_compute_binding(reinterpret_cast<Wrapper*>(rcx), N, rdx, r8, r9, s0);
    }
    if (K == list_kind && list_typed[N]) return list_typed[N](rcx, rdx, r8, r9, s0, s1, s2, s3);
    if (noop_census_enabled()) {
        const u64 args[8] = {rcx, rdx, r8, r9, s0, s1, s2, s3};
        noop_census_note(K, N, args, reinterpret_cast<u64>(_ReturnAddress()) - reinterpret_cast<u64>(GetModuleHandleW(nullptr)));
    }
    // Register arguments only: stale caller stack words often still point at an IID. The probe
    // costs over a millisecond per call in this process (memory queries), so only the first calls
    // of each slot are checked.
    static LONG shape_checks;
    if (shape_checks < 32 && x2w_local::first_calls(shape_checks, 32) && create_shaped(rdx, r8, r9, 0, 0, 0, 0)) {
        char name[64];
        std::snprintf(name, sizeof(name), "%s::slot%d(passes an IID)", kind_names[K], N);
        x2w_trap::fire(X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, name, N & 0xFF, rcx, rdx, r8, r9, _ReturnAddress());
    }
    static LONG calls;
    if (first_calls(calls, 1)) {
        char name[64];
        const char* known = K == list_kind ? x2w_list_slots::name(N) : nullptr;
        if (known) std::snprintf(name, sizeof(name), "%s::slot%d(%s)", kind_names[K], N, known);
        else std::snprintf(name, sizeof(name), "%s::slot%d", kind_names[K], N);
        x2w_trap::record("x2w-noop", X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, name, N & 0xFF, rcx, rdx, r8, r9,
                         _ReturnAddress());
    }
    return 0;
}

bool device_noop()
{
    char value[4]{};
    return GetEnvironmentVariableA("X2W_D3D12_DEVICE_NOOP", value, sizeof(value)) && value[0] == '1';
}

// With X2W_D3D12_DEVICE_NOOP=1: an unimplemented Xbox-order device method (slot 44 and up) logs its
// first call with the first 48 bytes behind rdx and r8 and returns 0.
template <int N>
__declspec(noinline) u64 device_noop_slot(u64 rcx, u64 rdx, u64 r8, u64 r9, u64 s0, u64 s1, u64 s2, u64 s3)
{
    Timed timed(time_device_noop);
    if (noop_census_enabled()) {
        const u64 args[8] = {rcx, rdx, r8, r9, s0, s1, s2, s3};
        noop_census_note(device_kind, N, args, reinterpret_cast<u64>(_ReturnAddress()) - reinterpret_cast<u64>(GetModuleHandleW(nullptr)));
    }
    static LONG shape_checks;
    if (shape_checks < 32 && x2w_local::first_calls(shape_checks, 32) && create_shaped(rdx, r8, r9, s0, s1, s2, s3)) {
        char name[64];
        std::snprintf(name, sizeof(name), "Device::slot%d(passes an IID)", N);
        char first[3 * 64 + 1];
        hex_text(reinterpret_cast<const void*>(rdx), 64, first, sizeof(first));
        log("x2w-d3d12 Device::slot%d passes an IID: rdx=0x%llx r8=0x%llx r9=0x%llx stack=0x%llx,0x%llx,0x%llx,0x%llx [rdx]=[%s]",
            N, rdx, r8, r9, s0, s1, s2, s3, first);
        x2w_trap::fire(X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, name, N & 0xFF, rcx, rdx, r8, r9, _ReturnAddress());
    }
    static LONG calls;
    if (first_calls(calls, 1)) {
        char name[64];
        std::snprintf(name, sizeof(name), "Device::slot%d", N);
        x2w_trap::record("x2w-noop", X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, name, N & 0xFF, rcx, rdx, r8, r9,
                         _ReturnAddress());
        char first[3 * 48 + 1], second[3 * 48 + 1];
        hex_text(reinterpret_cast<const void*>(rdx), 48, first, sizeof(first));
        hex_text(reinterpret_cast<const void*>(r8), 48, second, sizeof(second));
        log("x2w-noop   [rdx]=[%s] [r8]=[%s]", first, second);
    }
    return 0;
}

template <std::size_t... I>
void fill_device_noops(void** table, std::index_sequence<I...>)
{
    ((I >= 44 && table[I] == reinterpret_cast<void*>(&trap_slot<device_kind, static_cast<int>(I)>)
          ? (table[I] = reinterpret_cast<void*>(&device_noop_slot<static_cast<int>(I)>)) : nullptr), ...);
}

template <int K, std::size_t... I>
void fill_noops(void** table, std::index_sequence<I...>)
{
    ((I >= 8 ? (table[I] = reinterpret_cast<void*>(&noop_slot<K, static_cast<int>(I)>)) : nullptr), ...);
}

// With X2W_D3D12_DEVICE_NOOP=1, the same for the slots of the other object kinds that are still traps.
template <int K, std::size_t... I>
void fill_remaining_noops(void** table, std::index_sequence<I...>)
{
    ((I >= 8 && table[I] == reinterpret_cast<void*>(&trap_slot<K, static_cast<int>(I)>)
          ? (table[I] = reinterpret_cast<void*>(&noop_slot<K, static_cast<int>(I)>)) : nullptr), ...);
}
template <std::size_t... K>
void fill_all_remaining_noops(std::index_sequence<K...>)
{
    ((static_cast<int>(K) != device_kind && static_cast<int>(K) != list_kind
          ? fill_remaining_noops<static_cast<int>(K)>(tables[K], std::make_index_sequence<slot_count>{}) : void()), ...);
}

[[noreturn]] void fail(const char* what, const void* a, const void* b)
{
    x2w_trap::fire(X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, what, 0xFE, reinterpret_cast<u64>(a), reinterpret_cast<u64>(b),
                   0, 0, _ReturnAddress());
}

using Method = u64(__stdcall*)(IUnknown*, u64, u64, u64, u64, u64, u64, u64, u64);
Method method_of(IUnknown* real, int slot) { return reinterpret_cast<Method>((*reinterpret_cast<void***>(real))[slot]); }

// Desktop-order forwarder for a slot whose arguments are integers or pointers (no floating-point
// argument in the first four positions, at most eight arguments). Wrapped objects among the arguments
// are replaced by their desktop objects; unused argument words are caller stack the callee ignores.
template <int K, int N>
__declspec(noinline) u64 forward_slot(Wrapper* self, u64 a, u64 b, u64 c, u64 d, u64 e, u64 f, u64 g, u64 h)
{
    Timed timed(time_forward);
    const u64 result = method_of(self->own().real, N)(self->own().real, unwrap(a), unwrap(b), unwrap(c), unwrap(d), unwrap(e),
                                                unwrap(f), unwrap(g), unwrap(h));
    static LONG calls;
    if (first_calls(calls))
        log("x2w-d3d12 %s::slot%d(0x%llx, 0x%llx, 0x%llx, 0x%llx, 0x%llx) -> 0x%llx", kind_names[K], N, a, b, c, d, e,
            result);
    check_device(kind_names[K], N);
    return result;
}

// Desktop-order create method whose last two arguments are (riid, ppv), riid at argument index R
// (0-based, after the object). The created object is wrapped by its interface.
// Resource descriptions: the title sets flag bits desktop D3D12 rejects (0x20000 on buffers,
// 0x100000 and 0x200000 on render-target textures). Desktop receives only the bits it defines
// up to ALLOW_SIMULTANEOUS_ACCESS (0x3f); dropped bits are logged once per value.
// Console-only DXGI format values (116-120 in the console's format extensions: two 10-bit float
// formats, D16_UNORM_S8_UINT and its two typed views) mapped to the nearest desktop formats. The
// bit layouts differ, so contents are not interchangeable; this only lets the resources exist.
DXGI_FORMAT desktop_format(DXGI_FORMAT format)
{
    DXGI_FORMAT mapped = format;
    switch (static_cast<UINT>(format)) {
    case 116: case 117: mapped = DXGI_FORMAT_R10G10B10A2_UNORM; break;
    case 118: mapped = DXGI_FORMAT_D24_UNORM_S8_UINT; break;
    case 119: mapped = DXGI_FORMAT_R24_UNORM_X8_TYPELESS; break;
    case 120: mapped = DXGI_FORMAT_X24_TYPELESS_G8_UINT; break;
    default: break;
    }
    if (mapped != format) {
        static LONG calls;
        if (first_calls(calls, 6)) log("x2w-d3d12 console format %u -> desktop format %u", format, mapped);
    }
    return mapped;
}

// Depth-stencil formats are planar on desktop: a stencil view format reads plane 1, a depth one
// plane 0. A view with the wrong plane removes the device (DXGI_ERROR_INVALID_CALL).
void fix_srv_plane(D3D12_SHADER_RESOURCE_VIEW_DESC& d)
{
    const bool stencil = d.Format == DXGI_FORMAT_X32_TYPELESS_G8X24_UINT || d.Format == DXGI_FORMAT_X24_TYPELESS_G8_UINT;
    const bool depth = d.Format == DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS || d.Format == DXGI_FORMAT_R24_UNORM_X8_TYPELESS ||
                       d.Format == DXGI_FORMAT_R32_FLOAT || d.Format == DXGI_FORMAT_R16_UNORM;
    if (!stencil && !depth) return;
    const UINT plane = stencil ? 1 : 0;
    if (d.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2D) d.Texture2D.PlaneSlice = plane;
    else if (d.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2DARRAY) d.Texture2DArray.PlaneSlice = plane;
}

// LOCAL: a view of a multisampled resource must use a multisample dimension; the desktop removes the
// device (DXGI_ERROR_INVALID_CALL) for a TEXTURE2D/ARRAY view of one. The title's descriptions name
// the plain dimensions.
void fix_srv_samples(D3D12_SHADER_RESOURCE_VIEW_DESC& d, ID3D12Resource* resource)
{
    if (!resource) return;
    const D3D12_RESOURCE_DESC rd = resource->GetDesc();
    if (rd.SampleDesc.Count <= 1) return;
    if (d.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2D) {
        d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
        d.Texture2DMS = {};
    } else if (d.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2DARRAY) {
        const UINT first = d.Texture2DArray.FirstArraySlice, size = d.Texture2DArray.ArraySize;
        d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
        d.Texture2DMSArray = {};
        d.Texture2DMSArray.FirstArraySlice = first;
        d.Texture2DMSArray.ArraySize = size ? size : rd.DepthOrArraySize;
    }
}

// LOCAL: depth textures are created typeless on the desktop (typeless_depth_family), so a view has to
// name a format of that family. The console views its 16-bit depth with stencil (created here as
// R24G8) as R16_UNORM / R8_UINT and a 32-bit depth with stencil as R32_FLOAT; the desktop removes
// the device (DXGI_ERROR_INVALID_CALL) for those. Depth-named views read the depth plane, stencil-
// named views the stencil plane. Only the format changes; the bit layout of a 16-bit console depth
// view is not reproduced (it reads 24-bit depth as UNORM).
DXGI_FORMAT depth_view_format(DXGI_FORMAT resource, DXGI_FORMAT view)
{
    const bool stencil = view == DXGI_FORMAT_R8_UINT || view == DXGI_FORMAT_X24_TYPELESS_G8_UINT ||
                         view == DXGI_FORMAT_X32_TYPELESS_G8X24_UINT;
    switch (resource) {
    case DXGI_FORMAT_R24G8_TYPELESS:
        return stencil ? DXGI_FORMAT_X24_TYPELESS_G8_UINT : DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_R32G8X24_TYPELESS:
        return stencil ? DXGI_FORMAT_X32_TYPELESS_G8X24_UINT : DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case DXGI_FORMAT_R32_TYPELESS:
        return view == DXGI_FORMAT_D32_FLOAT || view == DXGI_FORMAT_UNKNOWN ? DXGI_FORMAT_R32_FLOAT : view;
    case DXGI_FORMAT_R16_TYPELESS:
        return view == DXGI_FORMAT_D16_UNORM || view == DXGI_FORMAT_UNKNOWN ? DXGI_FORMAT_R16_UNORM : view;
    default:
        return view;
    }
}

// The depth-stencil view format of a typeless depth family (DSVs need the D format).
DXGI_FORMAT depth_target_format(DXGI_FORMAT resource)
{
    switch (resource) {
    case DXGI_FORMAT_R24G8_TYPELESS: return DXGI_FORMAT_D24_UNORM_S8_UINT;
    case DXGI_FORMAT_R32G8X24_TYPELESS: return DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    case DXGI_FORMAT_R32_TYPELESS: return DXGI_FORMAT_D32_FLOAT;
    case DXGI_FORMAT_R16_TYPELESS: return DXGI_FORMAT_D16_UNORM;
    default: return DXGI_FORMAT_UNKNOWN;
    }
}

// A typed depth format of a texture that may be sampled -> its typeless family, so that shader
// resource views of it can exist on the desktop. Typed D formats allow no SRV at all.
DXGI_FORMAT typeless_depth_family(DXGI_FORMAT format)
{
    switch (format) {
    case DXGI_FORMAT_D24_UNORM_S8_UINT: return DXGI_FORMAT_R24G8_TYPELESS;
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return DXGI_FORMAT_R32G8X24_TYPELESS;
    case DXGI_FORMAT_D32_FLOAT: return DXGI_FORMAT_R32_TYPELESS;
    case DXGI_FORMAT_D16_UNORM: return DXGI_FORMAT_R16_TYPELESS;
    default: return format;
    }
}

// All shader-resource-view repairs for a view of `resource`: depth family format, plane, samples.
void fix_srv(D3D12_SHADER_RESOURCE_VIEW_DESC& d, ID3D12Resource* resource)
{
    if (!resource) return;
    const D3D12_RESOURCE_DESC rd = resource->GetDesc();
    if (depth_target_format(rd.Format) != DXGI_FORMAT_UNKNOWN) {
        const DXGI_FORMAT before = d.Format;
        d.Format = depth_view_format(rd.Format, d.Format);
        if (d.Format != before) {
            static LONG calls;
            if (first_calls(calls, 6)) log("x2w-d3d12 SRV of depth format %d: view format %d -> %d", rd.Format, before, d.Format);
        }
        fix_srv_plane(d);
    }
    fix_srv_samples(d, resource);
}

// A default shader resource view description for a typeless depth resource (a null description is
// refused for typeless resources).
bool default_depth_srv(ID3D12Resource* resource, D3D12_SHADER_RESOURCE_VIEW_DESC& d)
{
    if (!resource) return false;
    const D3D12_RESOURCE_DESC rd = resource->GetDesc();
    if (depth_target_format(rd.Format) == DXGI_FORMAT_UNKNOWN || rd.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) return false;
    d = {};
    d.Format = depth_view_format(rd.Format, DXGI_FORMAT_UNKNOWN);
    d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    if (rd.DepthOrArraySize > 1) {
        d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        d.Texture2DArray.MipLevels = UINT(-1);
        d.Texture2DArray.ArraySize = rd.DepthOrArraySize;
    } else {
        d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        d.Texture2D.MipLevels = UINT(-1);
    }
    fix_srv_samples(d, resource);
    return true;
}

// A default depth-stencil view description for a typeless depth resource.
bool default_depth_dsv(ID3D12Resource* resource, D3D12_DEPTH_STENCIL_VIEW_DESC& d)
{
    if (!resource) return false;
    const D3D12_RESOURCE_DESC rd = resource->GetDesc();
    const DXGI_FORMAT format = depth_target_format(rd.Format);
    if (format == DXGI_FORMAT_UNKNOWN || rd.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) return false;
    d = {};
    d.Format = format;
    const bool ms = rd.SampleDesc.Count > 1;
    if (rd.DepthOrArraySize > 1) {
        d.ViewDimension = ms ? D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY : D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
        if (ms) d.Texture2DMSArray.ArraySize = rd.DepthOrArraySize;
        else d.Texture2DArray.ArraySize = rd.DepthOrArraySize;
    } else {
        d.ViewDimension = ms ? D3D12_DSV_DIMENSION_TEXTURE2DMS : D3D12_DSV_DIMENSION_TEXTURE2D;
    }
    return true;
}

constexpr UINT desktop_resource_flags = 0x3f;
D3D12_RESOURCE_DESC sanitized(const D3D12_RESOURCE_DESC& desc)
{
    D3D12_RESOURCE_DESC copy = desc;
    const UINT dropped = static_cast<UINT>(desc.Flags) & ~desktop_resource_flags;
    if (dropped) {
        static UINT seen[16];
        static LONG count;
        bool known = false;
        for (LONG i = 0; i < count && i < 16; ++i) known |= seen[i] == dropped;
        if (!known && count < 16) {
            seen[count++] = dropped;
            log("x2w-d3d12 resource flags 0x%x: dropped Xbox bits 0x%x (dimension %d)", desc.Flags, dropped, desc.Dimension);
        }
    }
    copy.Flags = static_cast<D3D12_RESOURCE_FLAGS>(static_cast<UINT>(desc.Flags) & desktop_resource_flags);
    // Depth-stencil with unordered access or render target (seen: 0x6 on a depth texture) is refused by desktop.
    if (copy.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
        copy.Flags &= ~(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    // The Layout field carries a console swizzle mode (0x100 and up, the value XGSuggestSwizzleMode
    // works on); desktop gets UNKNOWN. A 4 KiB alignment request becomes the desktop default.
    if (static_cast<UINT>(desc.Layout) > D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE) {
        static LONG layouts;
        if (first_calls(layouts, 4)) log("x2w-d3d12 resource layout 0x%x (console swizzle) -> UNKNOWN", desc.Layout);
        copy.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    }
    if (desc.Alignment != 0 && desc.Alignment != 0x10000 && desc.Alignment != 0x400000) copy.Alignment = 0;
    copy.Format = desktop_format(desc.Format);
    if (x2w_gpu::gpu_abi_on()) copy.Format = x2w_srgb::backing_format(copy);
    if (desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && (copy.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) &&
        !(copy.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) && typeless_depth_family(copy.Format) != copy.Format) {
        static LONG depths;
        if (first_calls(depths, 4)) log("x2w-d3d12 depth texture format %d created typeless (%d) so it can be sampled", copy.Format, typeless_depth_family(copy.Format));
        copy.Format = typeless_depth_family(copy.Format);
    }
    // A packed target the adapter cannot render to or write as a UAV: packed_target_format.h.
    if (the_device && x2w_startup::packed_target_format(copy.Format)) {
        ID3D12Device* const device = static_cast<ID3D12Device*>(the_device->own().real);
        const DXGI_FORMAT stand_in = x2w_startup::packed_target_stand_in(copy, [device](DXGI_FORMAT format) -> UINT {
            D3D12_FEATURE_DATA_FORMAT_SUPPORT support{format};
            return SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support)))
                ? static_cast<UINT>(support.Support1) : 0u;
        });
        if (stand_in != copy.Format) {
            static LONG stand_ins;
            if (first_calls(stand_ins, 4))
                log("x2w-d3d12 texture format %d with flags 0x%x: shortcut - not supported for that use here, created as format %d",
                    copy.Format, copy.Flags, stand_in);
            copy.Format = stand_in;
        }
    }
    return copy;
}

// An optimized clear belongs to the actual texture format, including when the
// wrapper substitutes it. Preserve the caller's clear colour.
const D3D12_CLEAR_VALUE* packed_clear_value(const D3D12_CLEAR_VALUE* clear, DXGI_FORMAT requested,
                                          DXGI_FORMAT created, D3D12_CLEAR_VALUE& copy)
{
    const auto* mapping = x2w_startup::packed_target_mapping(requested);
    if (!clear || !mapping || created != mapping->replacement || desktop_format(clear->Format) != requested)
        return clear;
    copy = *clear;
    copy.Format = created;
    return &copy;
}

bool release_retired_enabled()
{
    static const bool on = x2w_local::env_on("X2W_D3D12_RELEASE_RETIRED");
    return on;
}

// X2W_D3D12_RELEASE_RETIRED_VIEWS=1 (needs X2W_D3D12_RELEASE_RETIRED=1): a released class member that only tracked slot-18
// descriptor slots keep alive is freed once every such slot has been replaced by a null descriptor.
bool release_retired_views_enabled()
{
    static const bool on = release_retired_enabled() && x2w_local::env_on("X2W_D3D12_RELEASE_RETIRED_VIEWS");
    return on;
}

// The retire class of X2W_D3D12_RELEASE_RETIRED, decided once at creation from the desktop description: a committed
// 2D texture with one sample, no render-target, depth-stencil or unordered-access use, in a default heap with no heap
// flags. Slot 27 and non-hooked slot 127 have no title memory; slot 133 committed 2D textures with title memory are
// included (cleaned from resource_address_index, texture_uploads and written_textures on retirement), while slot 127
// resources with __XMemAllocationHooks__ title memory remain excluded. Every use reaches the shim through handlers
// that retire_hold or shadow-slot SRV ref-counting cover, or that clear the flag; see
// docs/startup-runtime.md#d3d12-resource-budget.
RetireNever retire_classify(const D3D12_HEAP_PROPERTIES& props, D3D12_HEAP_FLAGS heap_flags, const D3D12_RESOURCE_DESC& desc,
                            const void* title_memory, bool allow_title_memory)
{
    if (desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) return RetireNever::buffer;
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) return RetireNever::dimension;
    if (desc.SampleDesc.Count != 1) return RetireNever::msaa;
    if ((desc.Flags & ~D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) != D3D12_RESOURCE_FLAG_NONE) return RetireNever::rt_ds_uav;
    if (props.Type != D3D12_HEAP_TYPE_DEFAULT) return RetireNever::heap_type;
    if (heap_flags != D3D12_HEAP_FLAG_NONE) return RetireNever::heap_flags;
    if (title_memory && !allow_title_memory) return RetireNever::title_memory;
    return RetireNever::none;
}

// Decides the retire class of a new wrapper once, from the desktop description, and records the reason for the diagnostics.
void retire_set_class(Wrapper* w, const D3D12_HEAP_PROPERTIES& props, D3D12_HEAP_FLAGS heap_flags,
                      const D3D12_RESOURCE_DESC& desc, const void* title_memory, bool allow_title_memory = false)
{
    if (!release_retired_enabled()) return;
    const RetireNever why = retire_classify(props, heap_flags, desc, title_memory, allow_title_memory);
    w->own().retire_never = static_cast<std::uint8_t>(why);
    w->own().retire_when_idle = why == RetireNever::none;
}

template <int K, int N, int R, int Desc = -1>
__declspec(noinline) u64 create_slot(Wrapper* self, u64 a, u64 b, u64 c, u64 d, u64 e, u64 f, u64 g, u64 h)
{
    Timed timed(time_create);
    u64 args[8] = {a, b, c, d, e, f, g, h};
    for (int i = 0; i < R; ++i) args[i] = unwrap(args[i]);
    D3D12_RESOURCE_DESC desc_copy{};
    D3D12_CLEAR_VALUE clear_copy{};
    DXGI_FORMAT requested_format = DXGI_FORMAT_UNKNOWN;
    if (Desc >= 0 && args[Desc]) {
        requested_format = desktop_format(reinterpret_cast<const D3D12_RESOURCE_DESC*>(args[Desc])->Format);
        desc_copy = sanitized(*reinterpret_cast<const D3D12_RESOURCE_DESC*>(args[Desc]));
        args[Desc] = reinterpret_cast<u64>(&desc_copy);
        if constexpr (K == device_kind && (N == 27 || N == 29))
            args[4] = reinterpret_cast<u64>(packed_clear_value(reinterpret_cast<const D3D12_CLEAR_VALUE*>(args[4]),
                                                               requested_format, desc_copy.Format, clear_copy));
    }
    const u64 result = method_of(self->own().real, N)(self->own().real, args[0], args[1], args[2], args[3], args[4], args[5],
                                                args[6], args[7]);
    check_device(kind_names[K], N);
    const auto* iid = reinterpret_cast<const GUID*>(args[R]);
    auto** out = reinterpret_cast<void**>(args[R + 1]);
    Kind kind = other_kind;
    if (SUCCEEDED(static_cast<HRESULT>(result)) && iid && out && *out) {
        kind = kind_for(*iid);
        if (Desc >= 0 && kind == resource_kind && requested_format != DXGI_FORMAT_UNKNOWN)
            x2w_startup::mark_packed_stand_in(static_cast<ID3D12Resource*>(*out), requested_format);
        Wrapper* child = wrap(static_cast<IUnknown*>(*out), kind);
        if constexpr (Desc >= 0) {
            if (kind == resource_kind && requested_format != desc_copy.Format && x2w_srgb::family(requested_format) == desc_copy.Format)
                child->own().srgb_uav_format = static_cast<unsigned char>(requested_format);
        }
        if constexpr (K == device_kind && N == 27) {
            if (kind == resource_kind && args[0])
                retire_set_class(child, *reinterpret_cast<const D3D12_HEAP_PROPERTIES*>(args[0]),
                                 static_cast<D3D12_HEAP_FLAGS>(args[1]), desc_copy, nullptr);
        }
        if constexpr (K == device_kind && (N == 29 || N == 30)) {
            if (kind == resource_kind && release_retired_enabled())
                child->own().retire_never = static_cast<std::uint8_t>(N == 29 ? RetireNever::aliased : RetireNever::reserved);
        }
        if constexpr (K == device_kind && (N == 27 || N == 28 || N == 29)) {
            if (x2w_budget::enabled()) {
                using Microsoft::WRL::ComPtr;
                auto* device = static_cast<ID3D12Device*>(self->own().real);
                if constexpr (N == 28) {
                    ComPtr<ID3D12Heap> heap;
                    if (SUCCEEDED(child->own().real->QueryInterface(IID_PPV_ARGS(&heap))))
                        child->own().budget = x2w_budget::backing_heap(heap.Get());
                } else {
                    ComPtr<ID3D12Resource> resource;
                    if (SUCCEEDED(child->own().real->QueryInterface(IID_PPV_ARGS(&resource)))) {
                        if constexpr (N == 27) {
                            const auto* props = reinterpret_cast<const D3D12_HEAP_PROPERTIES*>(a);
                            child->own().budget = x2w_budget::committed(device, resource.Get(), props->Type);
                        } else {
                            child->own().budget = x2w_budget::resource(device, resource.Get(), x2w_budget::placed);
                            if (auto* heap = as_wrapper(reinterpret_cast<const void*>(a))) {
                                child->own().backing_budget = heap->own().budget;
                                if (child->own().backing_budget) child->own().backing_budget->begin_use();
                            }
                        }
                    }
                }
            }
        }
        if (K == device_kind && kind == list_kind) {
            child->own().list_device = self;
            if (N == 12) child->own().list_node_mask = static_cast<UINT>(a);
        }
        *out = child;
    }
    static LONG calls;
    if (first_calls(calls, 4)) {
        char text[48];
        guid_text(iid, text, sizeof(text));
        log("x2w-d3d12 %s::slot%d create(0x%llx, 0x%llx, 0x%llx, 0x%llx) riid=%s -> 0x%08llx %s", kind_names[K], N, a, b,
            c, d, text, result, kind_names[kind]);
    }
    return result;
}

// Methods returning a structure by value: accept the object in either of the first two registers.
Wrapper* receiver(void* a, void* b, void** result, const char* method)
{
    // Struct-returning methods arrive as (return buffer, object) or (object, return buffer). A return
    // buffer on the caller's stack can still hold a stale copy of an object's vtable pointer and then
    // looks like a wrapper too: when both do, the one on this thread's stack is the buffer.
    if (as_wrapper(a) && as_wrapper(b)) {
        ULONG_PTR low = 0, high = 0;
        GetCurrentThreadStackLimits(&low, &high);
        const auto on_stack = [&](void* p) { return reinterpret_cast<ULONG_PTR>(p) >= low && reinterpret_cast<ULONG_PTR>(p) < high; };
        static LONG noted;
        if (first_calls(noted, 4)) log("x2w-d3d12 %s: both arguments look like objects (%p, %p); the stack one is the result", method, a, b);
        if (on_stack(a) && !on_stack(b)) {
            *result = a;
            return as_wrapper(b);
        }
    }
    if (Wrapper* w = as_wrapper(a)) {
        *result = b;
        return w;
    }
    *result = a;
    if (Wrapper* w = as_wrapper(b)) return w;
    fail(method, a, b);
}

// Slot 0, all kinds.
HRESULT __stdcall query_interface(Wrapper* self, const GUID* iid, void** out)
{
    char text[48];
    guid_text(iid, text, sizeof(text));
    if (!out || !iid) return E_POINTER;
    *out = nullptr;
    HRESULT hr = E_NOINTERFACE;
    if (is_base_iid(*iid) || kind_for(*iid) == self->own().kind) {
        const LONG refs = InterlockedIncrement(&self->own().refs);
        warm_texture(self, refs == 1);
        *out = self;
        hr = S_OK;
    }
    log("x2w-d3d12 %s::QueryInterface riid=%s -> 0x%08lx", kind_names[self->own().kind], text, static_cast<unsigned long>(hr));
    return hr;
}
ULONG __stdcall add_ref(Wrapper* self)
{
    const LONG refs = InterlockedIncrement(&self->own().refs);
    warm_texture(self, refs == 1);
    return static_cast<ULONG>(refs);
}
// The wrappers and their desktop objects stay alive for the process (experiment scope). The one exception is
// X2W_D3D12_RELEASE_RETIRED=1: the desktop object of a resource of the retire class is freed once nothing refers to it.
// The wrapper stays.
void retire_placement(Wrapper* w);
// LOCAL memcut: X2W_MEMCUT_RELEASE (d3d12_memcut_release.inl).
void memcut_released(Wrapper* self);
void memcut_list_reset(Wrapper* list);
void memcut_reap();
// LOCAL memcut: upload ring chunk lifetimes (d3d12_upload_ring.inl).
void ring_list_reset(Wrapper* list);
void ring_list_close(Wrapper* list);
void ring_list_gone(Wrapper* list);
void ring_submitted(const std::vector<Wrapper*>& lists);
void memcut_revive(Wrapper* w);
ID3D12Heap* memcut_slab_heap(void* address, UINT64 bytes, UINT64* offset);  // X2W_MEMCUT_ALIAS_SLABS
void memcut_slab_failed(void* address, UINT64 bytes, HRESULT hr);
void register_retire_queue(Wrapper* queue);
void retire_resource(Wrapper* w);
void retire_leave_class(const void* object, RetireLeft why);
void retire_add_view_ref(const void* object);
void retire_release_view_ref(const void* object);
void retire_drop_heap_views(Wrapper* heap);
void retire_note_title_release(Wrapper* w);
void retire_hold(Wrapper* list, Wrapper* resource);
void retire_drop_recording(Wrapper* list);
void retire_discard_holds(Wrapper* list);
void retire_note_submission(ID3D12CommandQueue* queue, const std::vector<Wrapper*>& lists);
void drain_retired_resources();
bool retire_reasons_enabled();
void retire_diag_note_release(Wrapper* w);
void drop_mesh_lease_owner(Wrapper* w);
void discard_mesh_lease_recording(ID3D12CommandList* list);
void clear_unplaced_buffers(Wrapper* device);
ULONG __stdcall release(Wrapper* self)
{
    const auto refs = InterlockedDecrement(&self->own().refs);
    if (!refs) {
        cool_texture(self);
        if (self->own().budget) self->own().budget->end_use();
        if (self->own().backing_budget) self->own().backing_budget->end_use();
        if (self->own().kind == descriptor_heap_kind) retire_drop_heap_views(self);
        if (self->own().kind == resource_kind && self->own().title_memory) {
            drop_mesh_lease_owner(self);
            retire_placement(self);
            if (self->own().retire_when_idle) retire_note_title_release(self);
        }
        if (self->own().kind == list_kind && self->own().real) {
            discard_mesh_lease_recording(static_cast<ID3D12CommandList*>(self->own().real));
            retire_drop_recording(self);
        }
        if (self->own().kind == resource_kind && retire_reasons_enabled()) retire_diag_note_release(self);
        if (self->own().retire_when_idle) retire_resource(self);
        if (self->own().kind == device_kind)
            clear_unplaced_buffers(self);
    }
    if (!refs && self->own().kind == resource_kind) memcut_released(self);  // LOCAL memcut
    if (!refs && self->own().kind == list_kind) ring_list_gone(self);       // LOCAL memcut (upload ring)
    if (!refs && x2w_gpu::gpu_abi_on() && (self->own().kind == resource_kind || self->own().gpu[0] || self->own().gpu[1]))
        gpu_released(self);
    return static_cast<ULONG>(refs);
}

// Slot 7 of every device child, desktop ID3D12DeviceChild::GetDevice(riid, ppv).
HRESULT __stdcall child_get_device(Wrapper* self, const GUID* iid, void** out)
{
    if (!out || !iid) return E_POINTER;
    return query_interface(the_device, iid, out);
}

ID3D12Device* device_of(Wrapper* w) { return static_cast<ID3D12Device*>(w->own().real); }

// Device slot 8, desktop CreateCommandQueue(desc, riid, ppv).
HRESULT __stdcall device_create_command_queue(Wrapper* self, const D3D12_COMMAND_QUEUE_DESC* desc, const GUID* iid,
                                              void** out)
{
    char text[48];
    guid_text(iid, text, sizeof(text));
    if (!out || !desc || !iid) return E_POINTER;
    *out = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    const HRESULT hr = kind_for(*iid) == queue_kind
        ? device_of(self)->CreateCommandQueue(desc, IID_PPV_ARGS(&queue)) : E_NOINTERFACE;
    log("x2w-d3d12 Device::CreateCommandQueue type=%d priority=%d flags=0x%x node=0x%x riid=%s -> 0x%08lx", desc->Type,
        desc->Priority, desc->Flags, desc->NodeMask, text, static_cast<unsigned long>(hr));
    if (FAILED(hr)) return hr;
    *out = wrap(queue, queue_kind);
    register_retire_queue(static_cast<Wrapper*>(*out));
    return S_OK;
}

// Device slot 48 (Xbox order; desktop slot 48 is OpenExistingHeapFromAddress): called four times at
// startup with a 16-byte queue descriptor (type 2, priority 0, flags 1/2/3/0, node 0/0/0/1), the
// desktop ID3D12CommandQueue IID and an output; the title asserts success. The Xbox flag meaning is
// unknown; the desktop queue is created with flags 0.
HRESULT __stdcall device_create_command_queue_x(Wrapper* self, const D3D12_COMMAND_QUEUE_DESC* desc, const GUID* iid,
                                                void** out)
{
    char text[48];
    guid_text(iid, text, sizeof(text));
    if (!out || !desc || !iid) return E_POINTER;
    *out = nullptr;
    D3D12_COMMAND_QUEUE_DESC copy = *desc;
    copy.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    ID3D12CommandQueue* queue = nullptr;
    const HRESULT hr = kind_for(*iid) == queue_kind
        ? device_of(self)->CreateCommandQueue(&copy, IID_PPV_ARGS(&queue)) : E_NOINTERFACE;
    log("x2w-d3d12 Device::slot48(CreateCommandQueue?) type=%d priority=%d flags=0x%x(dropped) node=0x%x riid=%s -> 0x%08lx",
        desc->Type, desc->Priority, desc->Flags, desc->NodeMask, text, static_cast<unsigned long>(hr));
    if (FAILED(hr)) return hr;
    *out = wrap(queue, queue_kind);
    register_retire_queue(static_cast<Wrapper*>(*out));
    return S_OK;
}

// Device slot 10/11, desktop CreateGraphics/ComputePipelineState(desc, riid, ppv): logged and refused
// until the Xbox shader bytecode is translated (the title's shaders are not DXIL).
template <int N>
HRESULT __stdcall device_create_pipeline(Wrapper*, const void* desc, const GUID* iid, void** out)
{
    char text[48];
    guid_text(iid, text, sizeof(text));
    char bytes[3 * 64 + 1];
    hex_text(desc, 64, bytes, sizeof(bytes));
    log("x2w-d3d12 Device::slot%d(Create%sPipelineState) desc=[%s] riid=%s -> trap", N, N == 10 ? "Graphics" : "Compute",
        bytes, text);
    if (out) *out = nullptr;
    trap_slot<device_kind, N>(0, reinterpret_cast<u64>(desc), reinterpret_cast<u64>(iid), reinterpret_cast<u64>(out));
    return E_NOTIMPL;
}

void note_console_size(const D3D12_RESOURCE_DESC& desc, const D3D12_RESOURCE_ALLOCATION_INFO& info)
{
    static LONG calls;
    if (first_calls(calls, 8))
        log("x2w-d3d12 allocation info: console size 0x%llx for %llux%u format %d samples %u (shortcut: desktop is larger)",
            info.SizeInBytes, desc.Width, desc.Height, desc.Format, desc.SampleDesc.Count);
}

// A single small 2D texture asked with a 4 KiB alignment gets the console's 4 KiB answer (small_texture_4k.h).
bool small_texture_answer(UINT count, const D3D12_RESOURCE_DESC* descs, D3D12_RESOURCE_ALLOCATION_INFO& info)
{
    if (count != 1 || !descs || !x2w_startup::small_texture_4k(descs[0], info)) return false;
    static LONG calls;
    if (first_calls(calls, 8))
        log("x2w-d3d12 allocation info: 4 KiB answer size 0x%llx for %llux%u mips %u format %d", info.SizeInBytes, descs[0].Width,
            descs[0].Height, descs[0].MipLevels, descs[0].Format);
    return true;
}

// Device slot 25, desktop GetResourceAllocationInfo(visibleMask, count, descs), returned by value.
void* __stdcall device_resource_allocation_info(void* a, void* b, UINT mask, UINT count, const D3D12_RESOURCE_DESC* descs)
{
    void* result = nullptr;
    Wrapper* self = receiver(a, b, &result, "Device::GetResourceAllocationInfo(no receiver)");
    std::vector<D3D12_RESOURCE_DESC> copies(count);
    for (UINT i = 0; i < count; ++i) copies[i] = sanitized(descs[i]);
    D3D12_RESOURCE_ALLOCATION_INFO info = device_of(self)->GetResourceAllocationInfo(mask, count, copies.data());
    // A single description of a family that is larger on desktop gets its console size (console_plane_size.h).
    if (count == 1 && x2w_startup::console_plane_size(descs[0], info)) note_console_size(descs[0], info);
    small_texture_answer(count, descs, info);
    *static_cast<D3D12_RESOURCE_ALLOCATION_INFO*>(result) = info;
    static LONG calls, invalid;
    if (first_calls(calls))
        log("x2w-d3d12 Device::GetResourceAllocationInfo count=%u dim=%d -> size=0x%llx align=0x%llx", count,
            count && descs ? descs[0].Dimension : -1, info.SizeInBytes, info.Alignment);
    if (info.SizeInBytes == ~0ull && first_calls(invalid, 8)) {
        for (UINT i = 0; i < count && i < 4; ++i) {
            char bytes[3 * 0x38 + 1];
            hex_text(&descs[i], 0x38, bytes, sizeof(bytes));
            log("x2w-d3d12   invalid desc[%u]=[%s]", i, bytes);
        }
    }
    return result;
}

// Device slot 38, desktop GetCopyableFootprints(desc, first, count, base, layouts, rows, row bytes, total).
void __stdcall device_copyable_footprints(Wrapper* self, const D3D12_RESOURCE_DESC* desc, UINT first, UINT count,
                                          UINT64 base, D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layouts, UINT* rows,
                                          UINT64* row_bytes, UINT64* total)
{
    const D3D12_RESOURCE_DESC copy = sanitized(*desc);
    device_of(self)->GetCopyableFootprints(&copy, first, count, base, layouts, rows, row_bytes, total);
}

// LOCAL: command signatures by wrapper: stride and the argument descriptors, for ExecuteIndirect.
struct SignatureInfo {
    UINT stride = 0;
    std::vector<D3D12_INDIRECT_ARGUMENT_DESC> args;
};
SRWLOCK signature_lock = SRWLOCK_INIT;
std::unordered_map<const void*, SignatureInfo>& signature_infos()
{
    static std::unordered_map<const void*, SignatureInfo> infos;
    return infos;
}
// Device slot 41, desktop CreateCommandSignature(desc, root, riid, ppv); on failure the descriptor and
// its argument descriptors are logged. The observed console-only type 0x69 gets a placeholder before
// desktop validation. Other types still reach the desktop even when they postdate DISPATCH_MESH.
u64 __stdcall device_create_command_signature(Wrapper* self, const D3D12_COMMAND_SIGNATURE_DESC* desc, u64 root,
                                              u64 iid, u64 out)
{
    // Argument types outside the desktop enumeration (0x69 seen, stride 12) cannot be created on
    // desktop and are not passed to it (the debug layer reports CREATECOMMANDSIGNATURE_INVALID).
    bool console_type = false;
    for (UINT i = 0; desc && i < desc->NumArgumentDescs; ++i)
        console_type |= static_cast<UINT>(desc->pArgumentDescs[i].Type) > D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;
    const u64 result = console_type ? static_cast<u64>(static_cast<UINT>(E_INVALIDARG))
                                    : create_slot<device_kind, 41, 2>(self, reinterpret_cast<u64>(desc), root, iid, out, 0, 0, 0, 0);
    void** output = reinterpret_cast<void**>(out);
    const bool readable_args = x2w_local::readable(desc, sizeof(*desc)) &&
        x2w_local::readable(desc->pArgumentDescs, std::size_t{desc->NumArgumentDescs} * sizeof(*desc->pArgumentDescs));
    if (SUCCEEDED(static_cast<HRESULT>(result)) && out && desc && root && desc->NumArgumentDescs > 1) {
        // A Dispatch preceded by root constants for the signature's root (#1759 part 2). Other root argument
        // types (root views, vertex and index buffer views) and graphics commands remain unsupported here.
        auto* signature = *reinterpret_cast<Wrapper**>(out);
        UINT bytes = sizeof(D3D12_DISPATCH_ARGUMENTS);
        bool constants_only = desc->pArgumentDescs[desc->NumArgumentDescs - 1].Type == D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
        for (UINT i = 0; constants_only && i + 1 < desc->NumArgumentDescs; ++i) {
            constants_only = desc->pArgumentDescs[i].Type == D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
            bytes += desc->pArgumentDescs[i].Constant.Num32BitValuesToSet * sizeof(UINT);
        }
        if (constants_only && desc->ByteStride == bytes && signature && signature->own().kind == command_signature_kind) {
            auto& state = signature->own();
            state.indirect_command = IndirectCommand::dispatch;
            state.signature_root = reinterpret_cast<ID3D12RootSignature*>(unwrap(root));
            state.indirect_bytes = bytes;
            state.signature_device = self;
        }
    }
    if (SUCCEEDED(static_cast<HRESULT>(result)) && out && desc && !root && desc->NumArgumentDescs == 1) {
        auto* signature = *reinterpret_cast<Wrapper**>(out);
        if (signature && signature->own().kind == command_signature_kind) {
            auto& state = signature->own();
            const auto type = desc->pArgumentDescs[0].Type;
            if (type == D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH && desc->ByteStride == sizeof(D3D12_DISPATCH_ARGUMENTS))
                state.indirect_command = IndirectCommand::dispatch;
            if (type == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW && desc->ByteStride == sizeof(D3D12_DRAW_ARGUMENTS))
                state.indirect_command = IndirectCommand::draw;
            if (type == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED && desc->ByteStride == sizeof(D3D12_DRAW_INDEXED_ARGUMENTS))
                state.indirect_command = IndirectCommand::draw_indexed;
            if (type == D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH && desc->ByteStride == sizeof(D3D12_DISPATCH_MESH_ARGUMENTS))
                state.indirect_command = IndirectCommand::dispatch_mesh;
            state.indirect_bytes = desc->ByteStride;
            if (state.indirect_command != IndirectCommand::none) state.signature_device = self;
        }
    }
    static LONG failures;
    if (FAILED(static_cast<HRESULT>(result)) && readable_args && first_calls(failures, 8)) {
        log("x2w-d3d12   CreateCommandSignature failed: stride=%u args=%u node=0x%x root=0x%llx", desc->ByteStride,
            desc->NumArgumentDescs, desc->NodeMask, root);
        for (UINT i = 0; i < desc->NumArgumentDescs && i < 8; ++i) {
            char bytes[3 * 16 + 1];
            hex_text(&desc->pArgumentDescs[i], sizeof(D3D12_INDIRECT_ARGUMENT_DESC), bytes, sizeof(bytes));
            log("x2w-d3d12     arg[%u]=[%s]", i, bytes);
        }
    }
    // The title gets a placeholder CommandSignature for those (command lists are not executed).
    if (FAILED(static_cast<HRESULT>(result)) && console_type && out) {
        log("x2w-d3d12   CreateCommandSignature: console argument type -> placeholder");
        *reinterpret_cast<void**>(out) = wrap(nullptr, command_signature_kind);
    }
    if (out && desc && x2w_local::readable(desc, sizeof(*desc)) && *reinterpret_cast<void**>(out) &&
        (SUCCEEDED(static_cast<HRESULT>(result)) || console_type)) {
        SignatureInfo info;
        info.stride = desc->ByteStride;
        if (desc->NumArgumentDescs <= 16 && x2w_local::readable(desc->pArgumentDescs, desc->NumArgumentDescs * sizeof(D3D12_INDIRECT_ARGUMENT_DESC)))
            info.args.assign(desc->pArgumentDescs, desc->pArgumentDescs + desc->NumArgumentDescs);
        static LONG logged;
        if (first_calls(logged, 40)) {
            char text[256] = "";
            int at = 0;
            for (const auto& a : info.args)
                at += std::snprintf(text + at, sizeof(text) - at > 0 ? sizeof(text) - at : 0, " %u(%u,%u)", static_cast<UINT>(a.Type),
                                    a.Constant.RootParameterIndex, a.Constant.Num32BitValuesToSet);
            log("x2w-d3d12   command signature %p: stride %u, args%s", *reinterpret_cast<void**>(out), info.stride, text);
        }
        AcquireSRWLockExclusive(&signature_lock);
        signature_infos()[*reinterpret_cast<void**>(out)] = std::move(info);
        ReleaseSRWLockExclusive(&signature_lock);
    }
    if (FAILED(static_cast<HRESULT>(result)) && console_type && out) return S_OK;
    return result;
}

// Device slot 108 (Xbox order): (mask, count, descs, info1) returned by value, the desktop
// ID3D12Device4::GetResourceAllocationInfo1 shape. The caller asks with a 4 KiB alignment first and
// repeats with 0 when the result's alignment is not 4 KiB.
void* __stdcall device_resource_allocation_info1(void* a, void* b, UINT mask, UINT count, const D3D12_RESOURCE_DESC* descs,
                                                 D3D12_RESOURCE_ALLOCATION_INFO1* info1)
{
    void* result = nullptr;
    Wrapper* self = receiver(a, b, &result, "Device::slot108(no receiver)");
    std::vector<D3D12_RESOURCE_DESC> copies(count);
    for (UINT i = 0; i < count; ++i) copies[i] = sanitized(descs[i]);
    D3D12_RESOURCE_ALLOCATION_INFO info{~0ull, 0};
    ID3D12Device4* device4 = nullptr;
    if (SUCCEEDED(device_of(self)->QueryInterface(IID_PPV_ARGS(&device4)))) {
        info = device4->GetResourceAllocationInfo1(mask, count, copies.data(), info1);
        device4->Release();
    }
    if (count == 1 && x2w_startup::console_plane_size(descs[0], info)) {
        if (info1) *info1 = {0, info.Alignment, info.SizeInBytes};
        note_console_size(descs[0], info);
    }
    if (small_texture_answer(count, descs, info) && info1) *info1 = {0, info.Alignment, info.SizeInBytes};
    *static_cast<D3D12_RESOURCE_ALLOCATION_INFO*>(result) = info;
    static LONG calls, invalid;
    if (first_calls(calls))
        log("x2w-d3d12 Device::slot108(GetResourceAllocationInfo1?) count=%u dim=%d align=0x%llx info1=%p -> size=0x%llx "
            "align=0x%llx", count, count ? descs[0].Dimension : -1, count ? descs[0].Alignment : 0ull,
            static_cast<void*>(info1), info.SizeInBytes, info.Alignment);
    if (info.SizeInBytes == ~0ull && first_calls(invalid, 8)) {
        char bytes[3 * 0x38 + 1];
        hex_text(descs, 0x38, bytes, sizeof(bytes));
        log("x2w-d3d12   slot108 invalid desc[0]=[%s]", bytes);
    }
    return result;
}

// Device slot 26, desktop GetCustomHeapProperties(nodeMask, heapType), returned by value.
void* __stdcall device_custom_heap_properties(void* a, void* b, UINT node_mask, D3D12_HEAP_TYPE type)
{
    void* result = nullptr;
    Wrapper* self = receiver(a, b, &result, "Device::GetCustomHeapProperties(no receiver)");
    const D3D12_HEAP_PROPERTIES properties = device_of(self)->GetCustomHeapProperties(node_mask, type);
    *static_cast<D3D12_HEAP_PROPERTIES*>(result) = properties;
    static LONG calls;
    if (first_calls(calls))
        log("x2w-d3d12 Device::GetCustomHeapProperties order=%s node=0x%x type=%d -> type=%d page=%d pool=%d",
            result == a ? "ret,this" : "this,ret", node_mask, type, properties.Type, properties.CPUPageProperty,
            properties.MemoryPoolPreference);
    return result;
}

// Device slot 43, desktop GetAdapterLuid(), returned by value.

// Device slot 15, desktop GetDescriptorHandleIncrementSize(type). A type desktop has is forwarded as before. Any
// other value makes desktop answer 0 and the debug layer report an error (#870): the stand-in logs the value once
// and answers 0 itself, with no increment made up for it.
static_assert(x2w_startup::desktop_descriptor_heap_type_count == D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES);
UINT __stdcall device_descriptor_increment_size(Wrapper* self, UINT type)
{
    if (!x2w_startup::is_valid_desktop_descriptor_heap_type(type)) {
        static LONG calls;
        if (first_calls(calls, 1)) log("x2w-d3d12 Device::slot15 out-of-range heap type 0x%x -> 0", type);
        return 0;
    }
    return static_cast<UINT>(forward_slot<device_kind, 15>(self, type, 0, 0, 0, 0, 0, 0, 0));
}

// Device slots 34/35, desktop MakeResident/Evict(count, objects).
template <int N>
HRESULT __stdcall device_residency(Wrapper* self, UINT count, void* const* objects)
{
    std::vector<ID3D12Pageable*> real(count);
    for (UINT i = 0; i < count; ++i) {
        retire_leave_class(objects[i], RetireLeft::residency);  // no hold tracks the residency of a desktop object
        real[i] = reinterpret_cast<ID3D12Pageable*>(unwrap(reinterpret_cast<u64>(objects[i])));
    }
    return N == 34 ? device_of(self)->MakeResident(count, real.data()) : device_of(self)->Evict(count, real.data());
}

// ---------------------------------------------------------------------------------------------
// X2W_D3D12_RENDER=1: first stage of real rendering. Command lists are really recorded for the
// calls implemented below (reset, close, render-target clears), really submitted, fences complete
// when the desktop GPU finishes, and the presented surface is read back and shown in the window.
// Everything else on a list is still dropped.
void release_targets(Wrapper* self);
void end_open_queries(Wrapper* self, ID3D12GraphicsCommandList* list);
void reset_list_dynamic_state(Wrapper* self);
void discard_list_gpu_writes(Wrapper* self);
void commit_list_gpu_writes(Wrapper* self);
void discard_draw_sampler_heaps(ID3D12CommandList* list);
void discard_texture_readbacks(ID3D12CommandList* list);
void discard_address_uploads(ID3D12CommandList* list);
void use_as_target(Wrapper* self, ID3D12Resource* resource);
bool is_container(const void* p);
bool shader_dump_dir(wchar_t* out, DWORD size);
void write_file(const wchar_t* path, const void* data, std::size_t bytes);
bool render_mode()
{
    static const bool on = [] {
        char value[4]{};
        return GetEnvironmentVariableA("X2W_D3D12_RENDER", value, sizeof(value)) && value[0] == '1';
    }();
    return on;
}

// Render-target view handle (desktop) -> the resource it views, recorded when the view is created.
SRWLOCK target_lock = SRWLOCK_INIT;
struct ViewTarget {
    ID3D12Resource* resource;
    int slot;  // 20 RTV, 21 DSV
};
std::unordered_map<u64, ViewTarget>& view_targets()
{
    static std::unordered_map<u64, ViewTarget> map;
    return map;
}
void note_view_target(u64 handle, ID3D12Resource* resource, int slot)
{
    AcquireSRWLockExclusive(&target_lock);
    view_targets()[handle] = {resource, slot};
    ReleaseSRWLockExclusive(&target_lock);
}
// slot 0: either kind. A handle created as the other kind yields nothing.
ID3D12Resource* view_target(u64 handle, int slot = 0)
{
    AcquireSRWLockShared(&target_lock);
    const auto found = view_targets().find(handle);
    ID3D12Resource* resource = found == view_targets().end() || (slot && found->second.slot != slot) ? nullptr : found->second.resource;
    ReleaseSRWLockShared(&target_lock);
    return resource;
}

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
{
    x2w_gpu::transition_all(list, resource, from, to);
}

// List slot 148 (Xbox order): AllocateUploadMemory(size, alignment) -> a pointer the CPU writes and
// the GPU reads at the same address (console unified memory), released when the list's allocator is
// reset. Used by the ray-tracing builder, which fails ("Failed to allocate upload buffer") on null.
// LOCAL SHORTCUT: a bump arena of 64 MiB blocks below 1 TiB (x2w_low); blocks are never released.
LONG upload_allocations;
std::mutex builder_upload_lock;
std::map<u64, u64> builder_upload_spans;
std::vector<u64> builder_upload_blocks;
bool builder_memory_on()
{
    static const bool on = x2w_local::env_on("X2W_GPU_MEMORY_STORES");
    return on;
}
u64 __stdcall list_allocate_upload(Wrapper*, u64 size, u64 alignment)
{
    static std::mutex arena_lock;
    static unsigned char *cursor = nullptr, *limit = nullptr;
    constexpr std::size_t block = std::size_t{64} << 20;
    if (!size) size = 16;
    if (alignment < 16) alignment = 16;
    if ((alignment & (alignment - 1)) || size > block / 2) return 0;
    const std::lock_guard guard(arena_lock);
    auto aligned = [&] { return reinterpret_cast<unsigned char*>((reinterpret_cast<u64>(cursor) + alignment - 1) & ~(alignment - 1)); };
    if (!cursor || aligned() + size > limit) {
        auto* fresh = static_cast<unsigned char*>(x2w_low::alloc(block, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (!fresh) return 0;
        cursor = fresh;
        limit = fresh + block;
        static const bool cpu_pages = x2w_local::env_on("X2W_GPU_MEMORY_CPU_PAGES");
        if (builder_memory_on() || cpu_pages) {
            std::lock_guard<std::mutex> lock(builder_upload_lock);
            builder_upload_blocks.push_back(reinterpret_cast<u64>(fresh));
        }
        log("x2w-d3d12 upload arena block %p (64 MiB)", static_cast<void*>(fresh));
    }
    unsigned char* result = aligned();
    cursor = result + size;
    if (builder_memory_on()) {
        std::lock_guard<std::mutex> lock(builder_upload_lock);
        builder_upload_spans.emplace(reinterpret_cast<u64>(result), size);
    }
    if (x2w_local::first_calls(upload_allocations, 4))
        log("x2w-d3d12 CommandList::slot148(AllocateUploadMemory) size=0x%llx align=0x%llx -> %p", size, alignment,
            static_cast<void*>(result));
    return reinterpret_cast<u64>(result);
}
LONG draw_dump_frame();
void discard_red_producer(ID3D12GraphicsCommandList* list);
void cull_capture_discard(ID3D12GraphicsCommandList* list);  // LOCAL: X2W_CULL_INPUT_CAPTURE (cull_input_capture.inl)
void cull_capture_submitted(ID3D12CommandQueue* queue, ID3D12CommandList* const* lists, UINT count);
void cull_capture_flush();
x2w_startup::DrawReadbacks& draw_dumps()
{
    static x2w_startup::DrawReadbacks tracker;
    return tracker;
}

LONG render_resets, render_closes, render_clears, render_executes, render_reset_failures;
// OMSetRenderTargets calls that bound targets, and render target / depth handles that resolved to no view.
LONG render_target_binds, render_target_unresolved;

// A console recording survives its allocator's Reset; a desktop one is discarded with it, and
// resetting an allocator whose lists the GPU still runs is invalid. Each title allocator therefore
// stands for a pool of desktop allocators. A list records on the pool's current one. In render
// mode the title's Reset resets the current one only when no list holds a recording on it and
// every queue has passed its last submit; otherwise the title continues on an idle allocator from
// the pool, or a new one, and the busy one keeps its recordings. Wrappers are never freed, so
// their addresses are stable keys.
struct DesktopAllocator {
    ID3D12CommandAllocator* real;
    LONG lists;                                             // lists whose recording is on this allocator
    std::vector<std::pair<ID3D12Fence*, UINT64>> submits;  // per queue fence: the last submit's value
    // A submit whose completion fence could not be created or signalled: the GPU may still run it
    // and nothing would say when it ends, so the allocator is never reset or reused again.
    bool untracked;
    // A command list recorded on this allocator failed Close(). Do not open another list on it
    // until it is idle and Reset(); hardware drivers can fault if a second recording starts on
    // an allocator whose failed recording is still attached.
    bool close_failed;
    std::vector<std::unique_ptr<x2w_upload::Arena>> uploads;
    std::vector<x2w_startup::ComPtr<ID3D12Resource>> indexed_validation;
    // Rectangle-list argument scratch of the recordings on this allocator. Retained with the recording, so until every
    // queue that ran it has passed its fence (the allocator is reused only when idle).
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> scratches;
};
struct TitleAllocator {
    std::vector<DesktopAllocator*> pool;
    DesktopAllocator* current = nullptr;
    D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT;  // from the lists recorded on it
};
SRWLOCK allocator_lock = SRWLOCK_INIT;
std::unordered_map<Wrapper*, TitleAllocator> title_allocators;
std::unordered_map<Wrapper*, DesktopAllocator*> list_recordings;  // list -> allocator holding its recording
std::unordered_map<ID3D12CommandQueue*, std::pair<ID3D12Fence*, UINT64>> queue_submits;
LONG allocator_swaps, allocator_creates;

// allocator_lock held.
TitleAllocator& title_allocator(Wrapper* allocator)
{
    TitleAllocator& title = title_allocators[allocator];
    if (!title.current) {
        title.current = new DesktopAllocator{static_cast<ID3D12CommandAllocator*>(allocator->own().real), 0, {}, false, false};
        title.pool.push_back(title.current);
    }
    return title;
}

// allocator_lock held.
bool allocator_idle(const DesktopAllocator* allocator)
{
    if (allocator->lists || allocator->untracked) return false;
    for (const auto& submit : allocator->submits)
        if (submit.first->GetCompletedValue() < submit.second) return false;
    return true;
}

// List slot 148 (Xbox order), allocate CPU-writable upload memory for this recording.
void* __stdcall list_allocate_upload_pages(Wrapper* self, UINT64 bytes, UINT64 alignment)
{
    if (!bytes || !alignment || (alignment & (alignment - 1)) ||
        bytes > UINT64_MAX - (std::max)(alignment, UINT64{16})) return nullptr;
    x2w_upload::Allocation allocation{};
    AcquireSRWLockExclusive(&allocator_lock);
    auto& state = self->own();
    const auto found = list_recordings.find(self);
    if (state.list_open && state.list_device && found != list_recordings.end() && found->second) {
        try {
            if (!state.list_uploads) {
                auto arena = std::make_unique<x2w_upload::Arena>();
                found->second->uploads.push_back(std::move(arena));
                state.list_uploads = found->second->uploads.back().get();
            }
            allocation = state.list_uploads->allocate(device_of(state.list_device), bytes, alignment);
        } catch (const std::bad_alloc&) {
            // Refuse the allocation without invalidating any range already recorded.
        }
    }
    ReleaseSRWLockExclusive(&allocator_lock);
    return reinterpret_cast<void*>(allocation.cpu);
}

// The rectangle-list argument scratch of this recording: created on first use, owned by the recording's desktop
// allocator (so it outlives every queue fence of the recording) and never shared with another recording.
ID3D12Resource* recording_rect_scratch(Wrapper* self)
{
    ID3D12Resource* scratch = nullptr;
    AcquireSRWLockExclusive(&allocator_lock);
    auto& state = self->own();
    const auto found = list_recordings.find(self);
    if (state.list_open && state.list_device && found != list_recordings.end() && found->second) {
        if (!state.list_scratch) {
            try {
                auto created = x2w_startup::RectIndirectArguments::make_scratch(device_of(state.list_device));
                if (created) {
                    found->second->scratches.push_back(std::move(created));
                    state.list_scratch = found->second->scratches.back().Get();
                }
            } catch (const std::bad_alloc&) {
                // No scratch: the caller refuses the draw.
            }
        }
        scratch = state.list_scratch;
    }
    ReleaseSRWLockExclusive(&allocator_lock);
    return scratch;
}

// Fixture query for two closed recordings before either is submitted.
extern "C" __declspec(dllexport) BOOL x2w_d3d12_distinct_rect_scratch(void* first, void* second)
{
    const Wrapper* a = as_wrapper(first);
    const Wrapper* b = as_wrapper(second);
    if (!a || !b || a == b || a->own().kind != list_kind || b->own().kind != list_kind) return FALSE;
    AcquireSRWLockShared(&allocator_lock);
    const auto& x = a->own();
    const auto& y = b->own();
    const bool distinct = x.list_ready && y.list_ready && x.list_scratch && y.list_scratch &&
                          x.list_scratch != y.list_scratch;
    ReleaseSRWLockShared(&allocator_lock);
    return distinct;
}

x2w_upload::Allocation list_upload_range(Wrapper* self, UINT64 address, UINT64 bytes)
{
    AcquireSRWLockShared(&allocator_lock);
    const auto& state = self->own();
    const auto allocation = state.list_open && state.list_uploads ? state.list_uploads->range(address, bytes) : x2w_upload::Allocation{};
    ReleaseSRWLockShared(&allocator_lock);
    return allocation;
}
UINT64 list_upload_gpu(Wrapper* self, UINT64 address, UINT64 bytes) { return list_upload_range(self, address, bytes).gpu; }

// Fixture lookup. The full range must belong to this list's open recording.
extern "C" __declspec(dllexport) UINT64 x2w_d3d12_upload_gpu_address(void* list, const void* cpu, UINT64 bytes)
{
    Wrapper* self = as_wrapper(list);
    return self && self->own().kind == list_kind && cpu && bytes
        ? list_upload_gpu(self, reinterpret_cast<UINT64>(cpu), bytes) : 0;
}

// Allocator slot 8, desktop Reset().
HRESULT __stdcall allocator_reset(Wrapper* self)
{
    if (!render_mode()) return static_cast<ID3D12CommandAllocator*>(self->own().real)->Reset();
    AcquireSRWLockExclusive(&allocator_lock);
    TitleAllocator& title = title_allocator(self);
    DesktopAllocator* next = nullptr;
    HRESULT hr = S_OK;
    if (allocator_idle(title.current)) {
        next = title.current;
    } else {
        for (DesktopAllocator* spare : title.pool)
            if (spare != title.current && allocator_idle(spare)) { next = spare; break; }
        if (!next) {
            ID3D12CommandAllocator* created = nullptr;
            hr = device_of(the_device)->CreateCommandAllocator(title.type, IID_PPV_ARGS(&created));
            if (SUCCEEDED(hr)) {
                next = new DesktopAllocator{created, 0, {}, false, false};
                title.pool.push_back(next);
                InterlockedIncrement(&allocator_creates);
            }
        }
    }
    if (next) {
        hr = next->real->Reset();
        if (SUCCEEDED(hr)) {
            next->uploads.clear();
            next->indexed_validation.clear();
            next->scratches.clear();
            next->submits.clear();
            next->close_failed = false;
            if (next != title.current) InterlockedIncrement(&allocator_swaps);
            title.current = next;
        }
    }
    const auto pool = title.pool.size();
    ReleaseSRWLockExclusive(&allocator_lock);
    static LONG calls;
    if ((FAILED(hr) || pool > 1) && first_calls(calls, 4))
        log("x2w-render allocator %p Reset -> 0x%08lx, %zu desktop allocators", static_cast<void*>(self),
            static_cast<unsigned long>(hr), pool);
    return hr;
}

// X2W_TEST_SUBMIT_FENCE_FAIL=1 (fixtures only): treat every submit fence as failed.
bool submit_fence_fails()
{
    static const bool on = GetEnvironmentVariableW(L"X2W_TEST_SUBMIT_FENCE_FAIL", nullptr, 0) != 0;
    return on;
}

// draw_lock and allocator_lock held, after the lists were executed on queue. Each list's allocator holds a fence value
// the queue must pass before the allocator can be reset. If that fence cannot be created or
// signalled, the work is on the GPU all the same: the allocators are marked untracked instead.
void note_submits(ID3D12CommandQueue* queue, const std::vector<Wrapper*>& lists)
{
    auto& [fence, value] = queue_submits[queue];
    if (!fence && !submit_fence_fails()) {
        ID3D12Device* device = nullptr;
        if (SUCCEEDED(queue->GetDevice(IID_PPV_ARGS(&device)))) {
            if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) fence = nullptr;
            device->Release();
        }
    }
    if (fence && SUCCEEDED(queue->Signal(fence, value + 1))) {
        ++value;
        for (Wrapper* list : lists) {
            const auto found = list_recordings.find(list);
            if (found == list_recordings.end()) continue;
            auto& submits = found->second->submits;
            const auto same = std::find_if(submits.begin(), submits.end(), [&](const auto& s) { return s.first == fence; });
            if (same == submits.end()) submits.emplace_back(fence, value);
            else same->second = value;
        }
    } else {
        for (Wrapper* list : lists)
            if (const auto found = list_recordings.find(list); found != list_recordings.end()) found->second->untracked = true;
        static LONG failures;
        if (first_calls(failures, 2)) log("x2w-render submit fence for allocator reuse failed; the allocators are not reused");
    }
}

// Fixtures: how many OMSetRenderTargets calls bound targets, and how many render target or depth handles
// (also in ClearRenderTargetView) resolved to no view.
extern "C" __declspec(dllexport) void x2w_d3d12_render_target_stats(unsigned* binds, unsigned* unresolved)
{
    if (binds) *binds = static_cast<unsigned>(InterlockedCompareExchange(&render_target_binds, 0, 0));
    if (unresolved) *unresolved = static_cast<unsigned>(InterlockedCompareExchange(&render_target_unresolved, 0, 0));
}

// Fixtures: the number of desktop allocators a title allocator stands for (0 if it has none yet).
extern "C" __declspec(dllexport) unsigned x2w_d3d12_allocator_pool_size(void* allocator)
{
    Wrapper* a = as_wrapper(allocator);
    if (!a || a->own().kind != allocator_kind) return 0;
    AcquireSRWLockShared(&allocator_lock);
    const auto found = title_allocators.find(a);
    const unsigned size = found == title_allocators.end() ? 0 : static_cast<unsigned>(found->second.pool.size());
    ReleaseSRWLockShared(&allocator_lock);
    return size;
}

void forget_list_dispatches(ID3D12GraphicsCommandList* list);  // LOCAL, by execute_dispatch
namespace gpu_time {  // LOCAL: X2W_CX_GPU_TIME (d3d12_gpu_time.inl)
void dispatch(Wrapper* self, ID3D12GraphicsCommandList* list, u64 key, bool after);
void close(Wrapper* self, ID3D12GraphicsCommandList* list);
void submitted(ID3D12CommandQueue* queue, const std::vector<Wrapper*>& wrappers);
void reset(Wrapper* self);
void report();
// LOCAL stall: X2W_GPU_TIME_CLASSES=1, more classes and the recording span
enum Class { c_cx = 0, c_draw, c_c62, c_copy, c_list, c_count };
void mark(Wrapper* self, ID3D12GraphicsCommandList* list, int cls, u64 key, bool after, bool drop = false);
void list_begin(Wrapper* self, ID3D12GraphicsCommandList* list);
}  // namespace gpu_time
// List slot 139 (Xbox order): (allocator, 0, 0), the per-frame reset.
HRESULT __stdcall list_reset(Wrapper* self, void* allocator)
{
    Wrapper* a = as_wrapper(allocator);
    HRESULT hr = E_INVALIDARG;
    x2w_startup::ComPtr<ID3D12GraphicsCommandList> replaced;
    AcquireSRWLockExclusive(&allocator_lock);
    auto& state = self->own();
    auto* previous = static_cast<ID3D12GraphicsCommandList*>(state.real);
    if (a && a->own().kind == allocator_kind && previous) {
        TitleAllocator& title = title_allocator(a);
        const auto type = previous->GetType();
        DesktopAllocator* target = title.current;
        if (target->close_failed && title.type == type) {
            target = nullptr;
            for (DesktopAllocator* spare : title.pool) {
                if (spare != title.current && allocator_idle(spare) && SUCCEEDED(spare->real->Reset())) {
                    spare->uploads.clear();
                    spare->indexed_validation.clear();
                    spare->scratches.clear();
                    spare->submits.clear();
                    spare->close_failed = false;
                    target = spare;
                    break;
                }
            }
            if (!target) {
                ID3D12CommandAllocator* created = nullptr;
                hr = device_of(state.list_device)->CreateCommandAllocator(type, IID_PPV_ARGS(&created));
                if (SUCCEEDED(hr)) {
                    target = new DesktopAllocator{created, 0, {}, false, false};
                    title.pool.push_back(target);
                    InterlockedIncrement(&allocator_creates);
                }
            }
        }
        if (target) {
            if (state.list_close_failed) {
                ID3D12GraphicsCommandList* replacement = nullptr;
                hr = device_of(state.list_device)->CreateCommandList(state.list_node_mask, type, target->real,
                                                                   nullptr, IID_PPV_ARGS(&replacement));
                if (SUCCEEDED(hr)) {
                    // Keep the old object through recording cleanup. In-flight GPU work keeps
                    // its allocator and submitted descriptors through the existing fence tracking.
                    replaced.Attach(previous);
                    state.real = replacement;
                    state.list_close_failed = false;
                }
            } else {
                hr = previous->Reset(target->real, nullptr);
            }
        }
        if (SUCCEEDED(hr)) {
            if (target != title.current) {
                InterlockedIncrement(&allocator_swaps);
                title.current = target;
            }
            title.type = type;
            DesktopAllocator*& on = list_recordings[self];
            if (on) --on->lists;
            on = title.current;
            ++on->lists;
            state.list_uploads = nullptr;
            state.list_scratch = nullptr;
            state.plane_copy_scratch = nullptr;
            state.list_ready = false;
            state.list_open = true;
        }
    }
    ReleaseSRWLockExclusive(&allocator_lock);
    if (SUCCEEDED(hr)) {
        memcut_list_reset(self);  // LOCAL memcut
        ring_list_reset(self);    // LOCAL memcut (upload ring)
        forget_list_dispatches(static_cast<ID3D12GraphicsCommandList*>(self->own().real));
        gpu_time::reset(self);
        gpu_time::list_begin(self, static_cast<ID3D12GraphicsCommandList*>(self->own().real));  // LOCAL: X2W_GPU_TIME_CLASSES
        self->own().list_ready = false;
        reset_list_dynamic_state(self);  // upstream superset of the local reset_viewport_state
        discard_list_gpu_writes(self);
        discard_draw_sampler_heaps(previous);
        discard_address_uploads(previous);
        discard_texture_readbacks(previous);
        discard_mesh_lease_recording(previous);
        retire_drop_recording(self);
        if (draw_dump_frame()) draw_dumps().discard(previous);
        discard_red_producer(previous);
        cull_capture_discard(previous);
    }
    InterlockedIncrement(SUCCEEDED(hr) ? &render_resets : &render_reset_failures);
    static LONG failures;
    if (FAILED(hr) && first_calls(failures, 6))
        log("x2w-render list Reset failed 0x%08lx (allocator %p)", static_cast<unsigned long>(hr), allocator);
    return hr;
}

// List slot 9, desktop Close().
HRESULT __stdcall list_close(Wrapper* self)
{
    AcquireSRWLockExclusive(&allocator_lock);
    auto& state = self->own();
    if (!state.list_open) {
        ReleaseSRWLockExclusive(&allocator_lock);
        return S_OK;
    }
    auto* list = static_cast<ID3D12GraphicsCommandList*>(state.real);
    end_open_queries(self, list);
    release_targets(self);
    gpu_time::close(self, list);  // LOCAL: X2W_CX_GPU_TIME
    const HRESULT hr = list->Close();
    state.list_open = false;
    state.list_ready = SUCCEEDED(hr);
    state.list_close_failed = FAILED(hr);
    if (FAILED(hr)) {
        if (const auto found = list_recordings.find(self); found != list_recordings.end() && found->second)
            found->second->close_failed = true;
    }
    ReleaseSRWLockExclusive(&allocator_lock);
    // LOCAL memcut (upload ring): a failed Close is never executed; it holds its chunks until a Reset. Outside
    // allocator_lock: the ring takes it (shared) under its own lock.
    ring_list_close(self);
    InterlockedIncrement(&render_closes);
    if (FAILED(hr)) {
        discard_list_gpu_writes(self);
        discard_draw_sampler_heaps(list);
        discard_address_uploads(list);
        discard_texture_readbacks(list);
        discard_mesh_lease_recording(list);
        retire_drop_recording(self);
        if (draw_dump_frame()) draw_dumps().discard(list);
        discard_red_producer(list);
        cull_capture_discard(list);
        static LONG failures;
        if (first_calls(failures, 6)) log("x2w-render list Close failed 0x%08lx", static_cast<unsigned long>(hr));
    }
    return hr;
}

// LOCAL stall (X2W_FENCE_TIMES=1, off by default): for every queue Signal that completes through
// complete_when_reached, the time from our Signal to the desktop fence reaching the value (t1: a
// time-critical probe thread waits on the same value with its own event, in signal order) and from
// there to our store of the completed value that the title mwaits on (t2: the threadpool callback).
// Reported with the present split. The probe works through the signals in order, so a later value of a
// faster queue can be timed late ("probe behind": store before probe; counted, not in the means).
namespace fence_times {
struct Entry {
    ID3D12Fence* fence;
    Wrapper* wrapper;
    UINT64 value;
    LONG64 t0, t1, t2;
    volatile LONG stage;
};
bool timing()
{
    static const bool value = x2w_local::env_on("X2W_FENCE_TIMES");
    return value;
}
// X2W_FENCE_STORE=probe: the probe thread also stores the completed value as soon as its wait returns
// (the threadpool callback still stores it too; fence_reached only ever raises the value).
bool store_early()
{
    static const bool value = [] {
        char text[16]{};
        return GetEnvironmentVariableA("X2W_FENCE_STORE", text, sizeof(text)) && std::strcmp(text, "probe") == 0;
    }();
    return value;
}
bool on() { return timing() || store_early(); }
SRWLOCK lock = SRWLOCK_INIT;
std::deque<Entry*> entries;
HANDLE wake;
volatile LONG64 count, gpu_sum, gpu_max, store_sum, store_max, total_max, behind, timeouts;
LONG64 now()
{
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart;
}
void raise(volatile LONG64& slot, LONG64 value)
{
    for (LONG64 seen = slot; value > seen;) {
        const LONG64 previous = InterlockedCompareExchange64(&slot, value, seen);
        if (previous == seen) break;
        seen = previous;
    }
}
void finish(Entry* e)
{
    if (InterlockedIncrement(&e->stage) != 2) return;
    if (e->t1 <= e->t2) {
        InterlockedIncrement64(&count);
        InterlockedAdd64(&gpu_sum, e->t1 - e->t0);
        InterlockedAdd64(&store_sum, e->t2 - e->t1);
        raise(gpu_max, e->t1 - e->t0);
        raise(store_max, e->t2 - e->t1);
    } else {
        InterlockedIncrement64(&behind);
    }
    raise(total_max, e->t2 - e->t0);
    e->fence->Release();
    delete e;
}
DWORD WINAPI probe(void*)
{
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    SetThreadDescription(GetCurrentThread(), L"x2w fence probe");
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    for (;;) {
        Entry* e = nullptr;
        AcquireSRWLockExclusive(&lock);
        if (!entries.empty()) {
            e = entries.front();
            entries.pop_front();
        }
        ReleaseSRWLockExclusive(&lock);
        if (!e) {
            WaitForSingleObject(wake, 100);
            continue;
        }
        if (e->fence->GetCompletedValue() < e->value && SUCCEEDED(e->fence->SetEventOnCompletion(e->value, event)) &&
            WaitForSingleObject(event, 5000) != WAIT_OBJECT_0)
            InterlockedIncrement64(&timeouts);
        else if (store_early())
            fence_reached(e->wrapper, e->value);
        e->t1 = now();
        finish(e);
    }
}
Entry* start(Wrapper* wrapper, UINT64 value)
{
    if (!on()) return nullptr;
    auto* fence = static_cast<ID3D12Fence*>(wrapper->own().real);
    static const bool started = [] {
        wake = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        CloseHandle(CreateThread(nullptr, 0, probe, nullptr, 0, nullptr));
        return true;
    }();
    (void)started;
    fence->AddRef();
    auto* e = new Entry{fence, wrapper, value, now(), 0, 0, 0};
    AcquireSRWLockExclusive(&lock);
    entries.push_back(e);
    ReleaseSRWLockExclusive(&lock);
    SetEvent(wake);
    return e;
}
void stored(Entry* e)
{
    if (!e) return;
    e->t2 = now();
    finish(e);
}
// One line per present split window; the counters restart.
void report()
{
    if (!timing()) return;
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    const auto ms = [&](LONG64 t) { return static_cast<double>(t) * 1000.0 / static_cast<double>(f.QuadPart); };
    const LONG64 n = InterlockedExchange64(&count, 0), gs = InterlockedExchange64(&gpu_sum, 0), ss = InterlockedExchange64(&store_sum, 0);
    const LONG64 gm = InterlockedExchange64(&gpu_max, 0), sm = InterlockedExchange64(&store_max, 0), tm = InterlockedExchange64(&total_max, 0);
    const LONG64 late = InterlockedExchange64(&behind, 0), lost = InterlockedExchange64(&timeouts, 0);
    AcquireSRWLockShared(&lock);
    const std::size_t queued = entries.size();
    ReleaseSRWLockShared(&lock);
    log("x2w-fence times: %lld signals; our Signal -> desktop fence reached mean %.2f ms max %.2f; reached -> completed "
        "value stored (what the title mwaits on) mean %.3f ms max %.2f; Signal -> stored max %.2f ms; probe behind %lld, "
        "probe timeouts %lld, queued %zu", static_cast<long long>(n), n ? ms(gs) / n : 0.0, ms(gm), n ? ms(ss) / n : 0.0, ms(sm),
        ms(tm), static_cast<long long>(late), static_cast<long long>(lost), queued);
}
}  // namespace fence_times

// A fence signalled on a queue completes when the desktop GPU reaches it.
struct PendingSignal {
    Wrapper* fence;
    UINT64 value;
    HANDLE event;
    fence_times::Entry* timing;
};
void CALLBACK signal_reached(PTP_CALLBACK_INSTANCE, void* context, PTP_WAIT wait, TP_WAIT_RESULT)
{
    auto* pending = static_cast<PendingSignal*>(context);
    fence_reached(pending->fence, pending->value);
    fence_times::stored(pending->timing);
    CloseHandle(pending->event);
    delete pending;
    CloseThreadpoolWait(wait);
}
void complete_when_reached(Wrapper* fence, UINT64 value)
{
    auto* pending = new PendingSignal{fence, value, CreateEventW(nullptr, FALSE, FALSE, nullptr),
                                      fence_times::start(fence, value)};
    if (FAILED(static_cast<ID3D12Fence*>(fence->own().real)->SetEventOnCompletion(value, pending->event))) {
        fence_reached(fence, value);
        fence_times::stored(pending->timing);
        CloseHandle(pending->event);
        delete pending;
        return;
    }
    PTP_WAIT wait = CreateThreadpoolWait(&signal_reached, pending, nullptr);
    SetThreadpoolWait(wait, pending->event, nullptr);
}

// The presented surface, read back and reduced for the window (BGRA, top-down).
constexpr int shown_width = 960, shown_height = 540;
std::vector<std::uint32_t> shown_pixels;
LONG shown_frames;
SRWLOCK shown_lock = SRWLOCK_INIT;
char shown_note[160] = "no frame read back yet";

void write_file(const wchar_t* path, const void* data, std::size_t bytes);
#include "startup_stubs/frame_rgb_report.inl"

void reduce_presented(const D3D12_RESOURCE_DESC& desc, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& footprint, bool rgba, UINT64 total,
                      ID3D12Resource* readback, LONG paired_frame, ID3D12Resource* surface);

// LOCAL stall: the read-back copy is issued here (issue) and picked up by a later call once its fence
// has passed, instead of waiting for it (up to 3 s, on the presenting thread, every 6th frame). Present
// calls this with issue false on every frame, so a copy is picked up at the first present after the GPU
// finished it. While a copy is in flight no new one is issued (its allocator is still in use).
// X2W_READBACK_WAIT=1 restores the wait after each copy.
void read_back_presented(ID3D12CommandQueue* queue, ID3D12Resource* surface, bool issue = true, LONG paired_frame = 0)
{
    static ID3D12CommandAllocator* allocator;
    static ID3D12GraphicsCommandList* list;
    static ID3D12Fence* fence;
    static ID3D12Resource* readback;
    static UINT64 readback_bytes, fence_value;
    static HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    static const bool wait_for_copy = x2w_local::env_on("X2W_READBACK_WAIT");
    struct InFlight {
        ID3D12Resource* surface = nullptr;  // held until the copy is done
        D3D12_RESOURCE_DESC desc{};
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        bool rgba = false;
        UINT64 total = 0;
        LONG paired_frame = 0;
    };
    static InFlight in_flight;
    if (in_flight.surface) {
        if (fence->GetCompletedValue() < fence_value) {
            if (paired_frame) log("x2w-frame-pair present submit_frame=%ld skipped: readback in flight", paired_frame);
            if (issue) std::snprintf(shown_note, sizeof(shown_note), "readback still in flight");
            return;
        }
        reduce_presented(in_flight.desc, in_flight.footprint, in_flight.rgba, in_flight.total, readback, in_flight.paired_frame, in_flight.surface);
        in_flight.surface->Release();
        in_flight.surface = nullptr;
    }
    if (!issue || !queue || !surface) return;
    ID3D12Device* device = device_of(the_device);
    const D3D12_RESOURCE_DESC desc = surface->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0, total = 0;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &row_bytes, &total);
    const bool rgba = desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM || desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    const bool bgra = desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM || desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    if ((!rgba && !bgra) || total == 0 || total == ~0ull) {
        std::snprintf(shown_note, sizeof(shown_note), "presented surface format %d not shown yet", desc.Format);
        return;
    }
    HRESULT hr = S_OK;
    if (!allocator) {
        hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
        if (SUCCEEDED(hr)) hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&list));
        if (SUCCEEDED(hr)) hr = list->Close();
        if (SUCCEEDED(hr)) hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    }
    if (SUCCEEDED(hr) && (!readback || readback_bytes < total)) {
        if (readback) readback->Release();
        readback = nullptr;
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = total;
        buffer.Height = 1;
        buffer.DepthOrArraySize = 1;
        buffer.MipLevels = 1;
        buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                             IID_PPV_ARGS(&readback));
        if (SUCCEEDED(hr)) x2w_budget::committed(device, readback, D3D12_HEAP_TYPE_READBACK);
        readback_bytes = SUCCEEDED(hr) ? total : 0;
    }
    if (SUCCEEDED(hr)) hr = allocator->Reset();
    if (SUCCEEDED(hr)) hr = list->Reset(allocator, nullptr);
    if (FAILED(hr)) {
        std::snprintf(shown_note, sizeof(shown_note), "readback setup failed 0x%08lx", static_cast<unsigned long>(hr));
        return;
    }
    transition(list, surface, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION from{}, to{};
    from.pResource = surface;
    from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    to.pResource = readback;
    to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    to.PlacedFootprint = footprint;
    list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    transition(list, surface, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    list->Close();
    ID3D12CommandList* lists[] = {list};
    queue->ExecuteCommandLists(1, lists);
    queue->Signal(fence, ++fence_value);
    surface->AddRef();
    in_flight = {surface, desc, footprint, rgba, total, paired_frame};
    if (!wait_for_copy) return;
    fence->SetEventOnCompletion(fence_value, event);
    if (WaitForSingleObject(event, 3000) != WAIT_OBJECT_0) {
        std::snprintf(shown_note, sizeof(shown_note), "readback timed out");
        return;  // picked up by a later call once the fence passes
    }
    reduce_presented(desc, footprint, rgba, total, readback, paired_frame, surface);
    in_flight.surface->Release();
    in_flight.surface = nullptr;
}

void reduce_presented(const D3D12_RESOURCE_DESC& desc, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& footprint, bool rgba, UINT64 total,
                      ID3D12Resource* readback, LONG paired_frame, ID3D12Resource* surface)
{
    void* mapped = nullptr;
    const D3D12_RANGE range{0, static_cast<SIZE_T>(total)};
    if (FAILED(readback->Map(0, &range, &mapped))) return;
    if (paired_frame) report_frame_rgb("present", paired_frame, surface, mapped, desc, footprint);
    AcquireSRWLockExclusive(&shown_lock);
    shown_pixels.resize(shown_width * shown_height);
    std::uint64_t sum = 0;
    for (int y = 0; y < shown_height; ++y) {
        const auto* row = static_cast<const unsigned char*>(mapped) + footprint.Offset +
                          static_cast<UINT64>(y) * desc.Height / shown_height * footprint.Footprint.RowPitch;
        for (int x = 0; x < shown_width; ++x) {
            const unsigned char* texel = row + static_cast<UINT64>(x) * desc.Width / shown_width * 4;
            const unsigned r = rgba ? texel[0] : texel[2], g = texel[1], b = rgba ? texel[2] : texel[0];
            shown_pixels[y * shown_width + x] = (r << 16) | (g << 8) | b;
            sum += r + g + b;
        }
    }
    ReleaseSRWLockExclusive(&shown_lock);
    // X2W_D3D12_DUMP_SURFACE=<file>: every 40th read-back is also written at full resolution as a
    // 32-bit BMP (overwritten each time), for looking at detail the reduced overlay copy loses.
    {
        static wchar_t dump_path[MAX_PATH];
        static const bool dump = GetEnvironmentVariableW(L"X2W_D3D12_DUMP_SURFACE", dump_path, MAX_PATH) != 0;
        // A path containing %ld gets the read-back number and a picture reduced to at most 960 pixels
        // wide, written at every read-back (about every 2 s), so a run keeps a series of shots.
        static const bool numbered = dump && std::wcsstr(dump_path, L"%ld") != nullptr;
        if (dump && (numbered || (shown_frames % 4) == 3) && desc.Width <= 8192 && desc.Height <= 8192) {
            const UINT step = numbered && desc.Width > 960 ? static_cast<UINT>((desc.Width + 959) / 960) : 1;
            const UINT width = static_cast<UINT>(desc.Width) / step, height = desc.Height / step;
            std::vector<unsigned char> file(54 + std::size_t{width} * height * 4);
            const std::uint32_t header[] = {static_cast<std::uint32_t>(file.size()), 0, 54, 40, width, static_cast<std::uint32_t>(-static_cast<int>(height)),
                                            1 | (32u << 16), 0, width * height * 4, 2835, 2835, 0, 0};
            file[0] = 'B';
            file[1] = 'M';
            std::memcpy(file.data() + 2, header, sizeof(header));
            for (UINT y = 0; y < height; ++y) {
                const auto* row = static_cast<const unsigned char*>(mapped) + footprint.Offset + UINT64{y} * step * footprint.Footprint.RowPitch;
                unsigned char* out = file.data() + 54 + std::size_t{y} * width * 4;
                for (UINT x = 0; x < width; ++x) {
                    const unsigned char* texel = row + std::size_t{x} * step * 4;
                    out[x * 4 + 0] = rgba ? texel[2] : texel[0];
                    out[x * 4 + 1] = texel[1];
                    out[x * 4 + 2] = rgba ? texel[0] : texel[2];
                    out[x * 4 + 3] = 255;
                }
            }
            if (numbered) {
                wchar_t path[MAX_PATH];
                swprintf(path, MAX_PATH, dump_path, static_cast<long>(shown_frames));
                write_file(path, file.data(), file.size());
            } else {
                write_file(dump_path, file.data(), file.size());
            }
        }
    }
    const D3D12_RANGE none{0, 0};
    readback->Unmap(0, &none);
    const LONG count = InterlockedIncrement(&shown_frames);
    std::snprintf(shown_note, sizeof(shown_note), "surface %llux%u format %d, readback %ld, mean level %.2f/255",
                  desc.Width, desc.Height, desc.Format, count, sum / (3.0 * shown_width * shown_height));
    if (x2w_local::first_of(count, 3) || count % 5 == 0) log("x2w-render presented %s (frame %ld)", shown_note, frame_number);
}

// LOCAL: render targets drawn into during a traced frame, in first-use order. At the end of the
// traced frame each is copied back on the presenting queue (after the title's work) and summarised:
// the share of texels that are not all zero and the mean byte value, to find the first empty pass.
SRWLOCK traced_lock = SRWLOCK_INIT;
struct TracedTarget {
    ID3D12Resource* first;  // the target
    int second;             // drawn draws
    ID3D12Resource* snap;   // copy taken right after its last drawn draw (COPY_DEST), or null
    bool copied = false;
    LONG command = 0;
    UINT slot = 0;
    DXGI_FORMAT view = DXGI_FORMAT_UNKNOWN;  // format of the RTV that produced this snapshot
};
std::vector<TracedTarget>& traced_targets()
{
    static std::vector<TracedTarget> targets;
    return targets;
}
struct PlaneStates {
    D3D12_RESOURCE_STATES depth, stencil;
};
PlaneStates current_planes(Wrapper* self, ID3D12Resource* resource, D3D12_RESOURCE_STATES fallback);
void require_planes(Wrapper* self, ID3D12Resource* resource, D3D12_RESOURCE_STATES depth, D3D12_RESOURCE_STATES stencil);
void require_state(Wrapper* self, ID3D12Resource* resource, D3D12_RESOURCE_STATES state);
// `after`: the state the target goes back to after a snapshot copy (LOCAL meshrt: DEPTH_WRITE for a depth target).
void note_traced_target(Wrapper* self, ID3D12Resource* target, bool drawn,
                        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_RENDER_TARGET, bool bounded = false,
                        DXGI_FORMAT view = DXGI_FORMAT_UNKNOWN, LONG command = 0, UINT slot = 0)
{
    if (!target || (!command && gpu_trace_stage_key())) return;
    AcquireSRWLockExclusive(&traced_lock);
    auto& all = traced_targets();
    auto found = std::find_if(all.begin(), all.end(), [&](const auto& e) { return e.first == target; });
    if (found == all.end()) {
        if (bounded) {
            const auto desc = target->GetDesc();
            // Up to eight single-sample color snapshots at up to 64 MiB each per selected frame.
            if (all.size() >= 8 || desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1 ||
                device_of(the_device)->GetResourceAllocationInfo(0, 1, &desc).SizeInBytes > (64ull << 20)) {
                ReleaseSRWLockExclusive(&traced_lock);
                return;
            }
        }
        target->AddRef();
        all.push_back({target, 0, nullptr});
        found = all.end() - 1;
    }
    static const bool snapshots = x2w_local::env_on("X2W_D3D12_TRACE_TARGETS");
    if (drawn) ++found->second;
    if (drawn && snapshots) {
        found->view = view;
        D3D12_RESOURCE_DESC desc = target->GetDesc();
        if (!found->snap && desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && desc.SampleDesc.Count == 1) {
            desc.Flags &= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;  // LOCAL meshrt: a depth target's copy keeps its depth format legal
            const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
            if (FAILED(device_of(the_device)->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                                       nullptr, IID_PPV_ARGS(&found->snap))))
                found->snap = nullptr;
        }
        if (found->snap && self->own().list_open) {
            auto* list = static_cast<ID3D12GraphicsCommandList*>(self->own().real);
            const auto before = current_planes(self, target, after);
            require_state(self, target, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->CopyResource(found->snap, target);
            found->copied = true;
            found->command = command;
            found->slot = slot;
            require_planes(self, target, before.depth, before.stencil);
        }
    }
    ReleaseSRWLockExclusive(&traced_lock);
}
// LOCAL meshrt, fixtures: the targets noted in traced frames so far, first-use order, 4 words each (width, height,
// format, drawn draws); returns how many are noted.
extern "C" __declspec(dllexport) unsigned x2w_d3d12_traced_targets(unsigned* out, unsigned capacity)
{
    AcquireSRWLockShared(&traced_lock);
    const auto& all = traced_targets();
    for (unsigned i = 0; out && i < capacity && i < all.size(); ++i) {
        const D3D12_RESOURCE_DESC desc = all[i].first->GetDesc();
        out[4 * i] = static_cast<unsigned>(desc.Width);
        out[4 * i + 1] = desc.Height;
        out[4 * i + 2] = static_cast<unsigned>(desc.Format);
        out[4 * i + 3] = static_cast<unsigned>(all[i].second);
    }
    const unsigned count = static_cast<unsigned>(all.size());
    ReleaseSRWLockShared(&traced_lock);
    return count;
}
void summarise_traced_targets(ID3D12CommandQueue* queue, LONG request_id = 0);
// LOCAL meshrt, fixtures that never present: the end-of-frame read-back (x2w-frame target lines) on a queue of its own;
// empties the noted list as at a present.
extern "C" __declspec(dllexport) int x2w_d3d12_summarise_traced_targets()
{
    if (!the_device) return 0;
    ID3D12CommandQueue* queue = nullptr;
    const D3D12_COMMAND_QUEUE_DESC desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    if (FAILED(device_of(the_device)->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue)))) return 0;
    summarise_traced_targets(queue);
    queue->Release();
    return 1;
}
void report_texture_pixels(const char* label, const char* grid_label, const void* mapped, const D3D12_RESOURCE_DESC& desc,
                           const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& footprint, UINT rows, UINT64 row_bytes)
{
    const UINT texel = desc.Width ? static_cast<UINT>(row_bytes / desc.Width) : 0;
    std::uint64_t nonzero = 0, sum = 0, texels = 0;
    for (UINT y = 0; y < rows; y += 2) {
        const auto* row = static_cast<const unsigned char*>(mapped) + footprint.Offset + UINT64{y} * footprint.Footprint.RowPitch;
        for (UINT64 x = 0; x < desc.Width && texel; x += 2) {
            bool any = false;
            for (UINT b = 0; b < texel; ++b) {
                any |= row[x * texel + b] != 0;
                sum += row[x * texel + b];
            }
            nonzero += any;
            ++texels;
        }
    }
    log("%s: non-zero texels %.4f, mean byte %.2f", label,
        texels ? double(nonzero) / texels : 0.0, texels && texel ? double(sum) / (texels * texel) : 0.0);
    // LOCAL (scene): raw texels on a 3x3 grid (at 1/6, 3/6, 5/6 of width and height), so a flat colour
    // and where it sits can be read per target without an image dump.
    if (texel && texel <= 16 && rows) {
        char text[9 * 36 + 1]{};
        std::size_t used = 0;
        for (UINT gy = 1; gy < 6; gy += 2)
            for (UINT gx = 1; gx < 6; gx += 2) {
                const UINT y = static_cast<UINT>(UINT64{rows} * gy / 6);
                const UINT64 x = desc.Width * gx / 6;
                const auto* at = static_cast<const unsigned char*>(mapped) + footprint.Offset + UINT64{y} * footprint.Footprint.RowPitch + x * texel;
                if (used + 2 * texel + 2 < sizeof(text)) {
                    for (UINT b = 0; b < texel; ++b) used += std::snprintf(text + used, sizeof(text) - used, "%02x", at[b]);
                    used += std::snprintf(text + used, sizeof(text) - used, " ");
                }
            }
        log("%s texels 3x3 (rows top to bottom, raw bytes): %s", grid_label, text);
    }
}
void summarise_traced_targets(ID3D12CommandQueue* queue, LONG request_id)
{
    wchar_t directory[MAX_PATH]{};
    const auto length = GetEnvironmentVariableW(L"X2W_TRACE_TARGET_DIR", directory, MAX_PATH);
    std::filesystem::path output_dir;
    if (length && length < MAX_PATH) {
        output_dir = directory;
        if (request_id) {
            output_dir /= L"request-" + std::to_wstring(request_id);
            std::error_code error;
            if (!std::filesystem::create_directory(output_dir, error)) {
                log("x2w-frame scene request %ld needs a fresh output directory (error %d)", request_id, error.value());
                output_dir.clear();
            }
        }
    }
    AcquireSRWLockExclusive(&traced_lock);
    std::vector<TracedTarget> all;
    all.swap(traced_targets());
    ReleaseSRWLockExclusive(&traced_lock);
    ID3D12Device* device = device_of(the_device);
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Fence* fence = nullptr;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    HRESULT hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
    if (SUCCEEDED(hr)) hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&list));
    if (SUCCEEDED(hr)) hr = list->Close();
    if (SUCCEEDED(hr)) hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    UINT64 fence_value = 0;
    for (std::size_t n = 0; n < all.size() && SUCCEEDED(hr); ++n) {
        ID3D12Resource* target = all[n].copied ? all[n].snap : all[n].first;
        const D3D12_RESOURCE_STATES rest = all[n].copied ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COMMON;
        const D3D12_RESOURCE_DESC desc = target->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT rows = 0;
        UINT64 row_bytes = 0, total = 0;
        device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &row_bytes, &total);
        if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || total == 0 || total == ~0ull || total > (512ull << 20)) {
            log("x2w-frame target %zu %p fmt %d: not summarised", n, static_cast<void*>(target), desc.Format);
            continue;
        }
        ID3D12Resource* readback = nullptr;
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = total;
        buffer.Height = 1;
        buffer.DepthOrArraySize = 1;
        buffer.MipLevels = 1;
        buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                   IID_PPV_ARGS(&readback)))) continue;
        allocator->Reset();
        list->Reset(allocator, nullptr);
        transition(list, target, rest, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION from{}, to{};
        from.pResource = target;
        from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        to.pResource = readback;
        to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        to.PlacedFootprint = footprint;
        list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        transition(list, target, D3D12_RESOURCE_STATE_COPY_SOURCE, rest);
        list->Close();
        ID3D12CommandList* lists[] = {list};
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence, ++fence_value);
        fence->SetEventOnCompletion(fence_value, event);
        if (WaitForSingleObject(event, 5000) != WAIT_OBJECT_0) {
            log("x2w-frame target %zu: read-back timed out", n);
            readback->Release();
            break;
        }
        void* mapped = nullptr;
        const D3D12_RANGE range{0, static_cast<SIZE_T>(total)};
        if (SUCCEEDED(readback->Map(0, &range, &mapped))) {
            char label[160], grid_label[48];
            std::snprintf(grid_label, sizeof(grid_label), "x2w-frame target %zu", n);
            std::snprintf(label, sizeof(label), "%s %s %p %llux%u fmt %d draws %d", grid_label, all[n].copied ? "snapshot" : "end",
                          static_cast<void*>(all[n].first), desc.Width, desc.Height, desc.Format, all[n].second);
            if (all[n].command)
                log("x2w-frame target %zu last recorded copy after command #%ld RTV[%u] resource=%p view_fmt=%u",
                    n, all[n].command, all[n].slot, static_cast<void*>(all[n].first), UINT(all[n].view));
            report_texture_pixels(label, grid_label, mapped, desc, footprint, rows, row_bytes);
            const auto preview = all[n].view != DXGI_FORMAT_UNKNOWN ? all[n].view : desc.Format;
            if (!output_dir.empty() && x2w_draw_bmp::trace_supported(preview)) {
                const auto path = output_dir / (L"target-" + std::to_wstring(n) + L".bmp");
                FILE* file = nullptr;
                if (_wfopen_s(&file, path.c_str(), L"wbx") == 0) {
                    const auto* pixels = static_cast<const unsigned char*>(mapped) + footprint.Offset;
                    const bool written = x2w_draw_bmp::write(file, pixels, footprint.Footprint.Width, rows,
                                                              footprint.Footprint.RowPitch, preview);
                    const bool closed = std::fclose(file) == 0;
                    if (!written || !closed) {
                        DeleteFileW(path.c_str());
                        log("x2w-frame target %zu: image write failed", n);
                    } else log("x2w-frame target %zu BMP view fmt %d%s", n, preview,
                        preview == DXGI_FORMAT_R16G16_FLOAT ? " signed RG, zero at 128" : "");
                } else {
                    log("x2w-frame target %zu: could not open image output", n);
                }
            }
            const D3D12_RANGE none{0, 0};
            readback->Unmap(0, &none);
        }
        readback->Release();
    }
    for (auto& e : all) {
        e.first->Release();
        if (e.snap) e.snap->Release();
    }
    if (list) list->Release();
    if (allocator) allocator->Release();
    if (fence) fence->Release();
    CloseHandle(event);
}

// ---------------------------------------------------------------------------------------------
// Render stage 2: real pipelines and draws for the shaders that have a translation.
//
// Stream and descriptor graphics pipelines share the VS/PS translation cache (d3d12_draw.inl).
std::size_t readable_window(u64 address, std::size_t wanted);
#include "startup_stubs/d3d12_capture.inl"

struct RootParameter {
    D3D12_ROOT_PARAMETER_TYPE type;
    UINT shader_register, space;
    UINT constants_count{};
};
SRWLOCK root_lock = SRWLOCK_INIT;
std::unordered_map<const Wrapper*, std::vector<RootParameter>>& root_infos()
{
    static std::unordered_map<const Wrapper*, std::vector<RootParameter>> map;
    return map;
}
// The heap-indexing flags (x2w_startup::root_flags_heap_indexed) each title root signature was created with.
std::unordered_map<const Wrapper*, UINT>& root_heap_flag_map()
{
    static std::unordered_map<const Wrapper*, UINT> map;
    return map;
}
UINT root_heap_flags(const Wrapper* root)
{
    AcquireSRWLockShared(&root_lock);
    const auto found = root_heap_flag_map().find(root);
    const UINT flags = found == root_heap_flag_map().end() ? 0 : found->second;
    ReleaseSRWLockShared(&root_lock);
    return flags;
}
const std::vector<RootParameter>* root_info(const Wrapper* root)
{
    AcquireSRWLockShared(&root_lock);
    const auto found = root_infos().find(root);
    const std::vector<RootParameter>* info = found == root_infos().end() ? nullptr : &found->second;
    ReleaseSRWLockShared(&root_lock);
    return info;
}

// Device slot 16, desktop CreateRootSignature(node, blob, size, riid, ppv); the parameter list is
// kept for the translated shaders (register and space of each root descriptor), and the heap-indexing
// flags for command-list slot 34.
HRESULT __stdcall device_create_root_signature(Wrapper* self, UINT node, const void* blob, SIZE_T size, const GUID* iid, void** out)
{
    if (!out || !iid) return E_POINTER;
    *out = nullptr;
    ID3D12RootSignature* root = nullptr;
    // LOCAL: the title's root signatures carry CBV_SRV_UAV/SAMPLER_HEAP_DIRECTLY_INDEXED. The title's
    // SetDescriptorHeaps is not forwarded (translated draws bind their own heaps), so setting such a
    // root on a desktop list is an error (#1419). No title shader runs against these roots on the
    // desktop, so the two flags are cleared there; the title's blob is kept for translation keys.
    ID3DBlob* stripped = nullptr;
    {
        ID3D12VersionedRootSignatureDeserializer* reader = nullptr;
        if (blob && SUCCEEDED(D3D12CreateVersionedRootSignatureDeserializer(blob, size, IID_PPV_ARGS(&reader)))) {
            const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* desc = nullptr;
            constexpr UINT indexed = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED |
                                     D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;
            if (SUCCEEDED(reader->GetRootSignatureDescAtVersion(D3D_ROOT_SIGNATURE_VERSION_1_1, &desc)) && desc &&
                (static_cast<UINT>(desc->Desc_1_1.Flags) & indexed)) {
                D3D12_VERSIONED_ROOT_SIGNATURE_DESC copy = *desc;
                copy.Desc_1_1.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(static_cast<UINT>(copy.Desc_1_1.Flags) & ~indexed);
                if (FAILED(D3D12SerializeVersionedRootSignature(&copy, &stripped, nullptr))) stripped = nullptr;
                static LONG strips;
                if (first_calls(strips, 4))
                    log("x2w-d3d12 Device::CreateRootSignature flags 0x%x: heap-directly-indexed bits cleared for the desktop (%s)",
                        static_cast<UINT>(desc->Desc_1_1.Flags), stripped ? "ok" : "serialize failed, kept");
            }
            reader->Release();
        }
    }
    const HRESULT hr = kind_for(*iid) == root_signature_kind
        ? device_of(self)->CreateRootSignature(node, stripped ? stripped->GetBufferPointer() : blob,
                                               stripped ? stripped->GetBufferSize() : size, IID_PPV_ARGS(&root))
        : E_NOINTERFACE;
    if (stripped) stripped->Release();
    static LONG calls;
    const bool log_it = first_calls(calls, 6);
    if (FAILED(hr)) {
        log("x2w-d3d12 Device::CreateRootSignature size=0x%llx -> 0x%08lx", static_cast<u64>(size), static_cast<unsigned long>(hr));
        return hr;
    }
    Wrapper* w = wrap(root, root_signature_kind);
    note_root_blob(w, blob, size);
    std::vector<RootParameter> parameters;
    UINT heap_flags = 0;
    ID3D12VersionedRootSignatureDeserializer* deserializer = nullptr;
    if (SUCCEEDED(D3D12CreateVersionedRootSignatureDeserializer(blob, size, IID_PPV_ARGS(&deserializer)))) {
        const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* desc = nullptr;
        if (SUCCEEDED(deserializer->GetRootSignatureDescAtVersion(D3D_ROOT_SIGNATURE_VERSION_1_1, &desc)) && desc) {
            heap_flags = static_cast<UINT>(desc->Desc_1_1.Flags) & x2w_startup::root_flags_heap_indexed;
            w->own().root_without_bindings = !heap_flags && !desc->Desc_1_1.NumParameters && !desc->Desc_1_1.NumStaticSamplers;
            for (UINT i = 0; i < desc->Desc_1_1.NumParameters; ++i) {
                const auto& parameter = desc->Desc_1_1.pParameters[i];
                RootParameter info{parameter.ParameterType, 0, 0};
                if (parameter.ParameterType == D3D12_ROOT_PARAMETER_TYPE_CBV || parameter.ParameterType == D3D12_ROOT_PARAMETER_TYPE_SRV ||
                    parameter.ParameterType == D3D12_ROOT_PARAMETER_TYPE_UAV) {
                    info.shader_register = parameter.Descriptor.ShaderRegister;
                    info.space = parameter.Descriptor.RegisterSpace;
                } else if (parameter.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS) {
                    info.shader_register = parameter.Constants.ShaderRegister;
                    info.space = parameter.Constants.RegisterSpace;
                    info.constants_count = parameter.Constants.Num32BitValues;
                }
                parameters.push_back(info);
            }
        }
        deserializer->Release();
    }
    if (log_it) {
        char text[256] = "";
        int used = 0;
        for (std::size_t i = 0; i < parameters.size() && used < 220; ++i)
            used += std::snprintf(text + used, sizeof(text) - used, " %d:r%u,s%u", parameters[i].type, parameters[i].shader_register,
                                  parameters[i].space);
        log("x2w-d3d12 Device::CreateRootSignature size=0x%llx -> %p heap flags 0x%x parameters (type:register,space)%s",
            static_cast<u64>(size), static_cast<void*>(w), heap_flags, text);
    }
    AcquireSRWLockExclusive(&root_lock);
    root_infos()[w] = std::move(parameters);
    root_heap_flag_map()[w] = heap_flags;
    ReleaseSRWLockExclusive(&root_lock);
    *out = w;
    return S_OK;
}

// Fixtures: the copy of a root signature's blob kept for shader translation (the .root request and the
// translation key). Returns its size and copies up to capacity bytes to out; 0 for anything else. The
// heap-indexing flags slot 34 acts on (root_heap_flags, read from the desktop blob) go to heap_flags.
extern "C" __declspec(dllexport) std::size_t x2w_d3d12_root_translation_blob(void* root, void* out, std::size_t capacity, UINT* heap_flags)
{
    Wrapper* w = as_wrapper(root);
    if (!w || w->own().kind != root_signature_kind) return 0;
    if (heap_flags) *heap_flags = root_heap_flags(w);
    AcquireSRWLockShared(&note_lock);
    const auto found = root_blobs().find(w);
    const std::size_t bytes = found == root_blobs().end() ? 0 : found->second.size();
    if (bytes && out) std::memcpy(out, found->second.data(), bytes < capacity ? bytes : capacity);
    ReleaseSRWLockShared(&note_lock);
    return bytes;
}

bool readable_shader_container(const D3D12_SHADER_BYTECODE& code)
{
    return code.BytecodeLength >= 32 && code.BytecodeLength <= (1u << 20) &&
        x2w_local::readable(code.pShaderBytecode, code.BytecodeLength) &&
        !std::memcmp(code.pShaderBytecode, "DXBC", 4);
}

bool desktop_indirect_graphics_shader(const D3D12_SHADER_BYTECODE& code, D3D12_SHADER_VERSION_TYPE stage)
{
    if (!readable_shader_container(code)) return false;
    x2w_startup::ComPtr<ID3D12ShaderReflection> shader;
    if (FAILED(D3DReflect(code.pShaderBytecode, code.BytecodeLength, IID_PPV_ARGS(shader.GetAddressOf())))) return false;
    D3D12_SHADER_DESC desc{};
    if (FAILED(shader->GetDesc(&desc)) || desc.Version >> 16 != stage ||
        ((desc.Version >> 4) & 0xf) != 5 || desc.BoundResources) return false;
    if (stage == D3D12_SHVER_VERTEX_SHADER) {
        for (UINT i = 0; i < desc.InputParameters; ++i) {
            D3D12_SIGNATURE_PARAMETER_DESC input{};
            if (FAILED(shader->GetInputParameterDesc(i, &input)) ||
                (input.SystemValueType != D3D_NAME_VERTEX_ID && input.SystemValueType != D3D_NAME_INSTANCE_ID))
                return false;
        }
    }
    return true;
}

constexpr std::size_t stream_sizes[23] = {16, 24, 24, 24, 24, 24, 24, 40, 336, 8, 48, 56, 24, 8, 8, 40, 8, 16, 8, 24, 8, 64, 24};

enum class DxilMeshStage : unsigned { pixel = 0, mesh = 13, amplification = 14, unknown = 0xffff };

DxilMeshStage desktop_dxil_mesh_stage(const D3D12_SHADER_BYTECODE& code)
{
    if (!readable_shader_container(code)) return DxilMeshStage::unknown;
    const auto* data = static_cast<const unsigned char*>(code.pShaderBytecode);
    const auto word = [data](std::size_t offset) {
        UINT value;
        std::memcpy(&value, data + offset, sizeof(value));
        return value;
    };
    const std::size_t bytes = code.BytecodeLength;
    const std::size_t parts = word(28);
    if (word(24) != bytes || parts > (bytes - 32) / 4) return DxilMeshStage::unknown;
    const std::size_t header_size = 32 + parts * 4;
    DxilMeshStage stage = DxilMeshStage::unknown;
    bool found_dxil = false;
    for (std::size_t i = 0; i < parts; ++i) {
        const std::size_t part = word(32 + i * 4);
        if (part < header_size || part > bytes - 8) return DxilMeshStage::unknown;
        const std::size_t size = word(part + 4);
        if (size > bytes - part - 8) return DxilMeshStage::unknown;
        if (!std::memcmp(data + part, "XDAT", 4)) return DxilMeshStage::unknown;
        if (std::memcmp(data + part, "DXIL", 4)) continue;
        if (found_dxil || size < 24) return DxilMeshStage::unknown;
        found_dxil = true;
        const std::size_t program = part + 8;
        const std::size_t bitcode = word(program + 16);
        const std::size_t bitcode_size = word(program + 20);
        if (word(program + 4) * 4ull != size || std::memcmp(data + program + 8, "DXIL", 4) ||
            bitcode < 16 || bitcode > size - 8 || bitcode_size < 4 || bitcode_size > size - 8 - bitcode ||
            std::memcmp(data + program + 8 + bitcode, "BC\xc0\xde", 4)) return DxilMeshStage::unknown;
        const UINT version = word(program);
        if (((version >> 4) & 0xf) != 6) return DxilMeshStage::unknown;
        if (version >> 16 == static_cast<UINT>(DxilMeshStage::pixel)) stage = DxilMeshStage::pixel;
        if (version >> 16 == static_cast<UINT>(DxilMeshStage::mesh)) stage = DxilMeshStage::mesh;
        if (version >> 16 == static_cast<UINT>(DxilMeshStage::amplification)) stage = DxilMeshStage::amplification;
    }
    return stage;
}

std::size_t desktop_stream_size(UINT type)
{
    if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS || type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS) return 24;
    return type < std::size(stream_sizes) ? stream_sizes[type] : 0;
}

// Converts the supported stream layout to desktop form and creates the pipeline.
ID3D12PipelineState* build_stream_pipeline(const D3D12_PIPELINE_STATE_STREAM_DESC* desc, char* why, std::size_t why_size,
                                          bool& compute, IndirectGraphicsPipeline& graphics,
                                          ID3D12RootSignature* cached_root = nullptr)
{
    const bool cached_mesh = cached_root != nullptr;
    const std::size_t bytes = desc->SizeInBytes;
    if (bytes < 16 || bytes > 0x2000 || (bytes & 7) || !x2w_local::readable(desc->pPipelineStateSubobjectStream, bytes)) {
        std::snprintf(why, why_size, "stream not readable");
        return nullptr;
    }
    std::vector<u64> copy(bytes / 8);
    std::memcpy(copy.data(), desc->pPipelineStateSubobjectStream, bytes);
    auto* stream = reinterpret_cast<unsigned char*>(copy.data());
    const Wrapper* root = nullptr;
    bool vertex = false, pixel = false, mesh = false, input_layout = false;
    bool amplification = false, console_mesh = false;
    D3D12_PRIMITIVE_TOPOLOGY_TYPE topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED;
    D3D12_RT_FORMAT_ARRAY targets{};
    DXGI_FORMAT depth = DXGI_FORMAT_UNKNOWN;
    DXGI_SAMPLE_DESC samples{1, 0};
    bool depth_enabled = false, depth_writes = false, untracked_depth_state = false;
    bool ok = true;
    // First pass: validate the layout and find the wrapped root signature.
    for (std::size_t offset = 0; offset < bytes;) {
        auto& type = *reinterpret_cast<UINT*>(stream + offset);
        const auto size = desktop_stream_size(type);
        if (!size || size > bytes - offset) {
            ok = false;
            std::snprintf(why, why_size, "unknown or truncated subobject %u at +0x%zx", type, offset);
            break;
        }
        if (type == 0) root = as_wrapper(*reinterpret_cast<void**>(stream + offset + 8));
        if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS || type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS) {
            const auto& code = *reinterpret_cast<const D3D12_SHADER_BYTECODE*>(stream + offset + 8);
            if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS) {
                // Xbox MS and desktop AS share tag 24. Classify the actual desktop DXIL before rewriting.
                const auto stage = desktop_dxil_mesh_stage(code);
                if (stage == DxilMeshStage::mesh && !mesh && !amplification) {
                    type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS;
                    console_mesh = true;
                } else if (stage == DxilMeshStage::amplification && !amplification && !console_mesh) {
                    amplification = true;
                } else {
                    std::snprintf(why, why_size, "ambiguous or unsupported DXIL for shared mesh tag at +0x%zx", offset);
                    ok = false;
                    break;
                }
            }
            if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS) {
                if (mesh) {
                    std::snprintf(why, why_size, "duplicate mesh stage at +0x%zx", offset);
                    ok = false;
                    break;
                }
                mesh = true;
            }
        }
        offset += size;
    }
    const bool empty_root = root && root->own().kind == root_signature_kind && root->own().root_without_bindings;
    const bool native_mesh_root = empty_root || cached_mesh;
    for (std::size_t offset = 0; ok && offset < bytes;) {
        const UINT type = *reinterpret_cast<UINT*>(stream + offset);
        if (type == 0) {
            u64& pointer = *reinterpret_cast<u64*>(stream + offset + 8);
            pointer = cached_mesh ? reinterpret_cast<u64>(cached_root) : unwrap(pointer);
        } else if (type >= 1 && type <= 5) {
            const auto* code = reinterpret_cast<const D3D12_SHADER_BYTECODE*>(stream + offset + 8);
            if (code->pShaderBytecode) {
                const bool is_vertex = type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS;
                bool supported = false;
                if ((mesh ? native_mesh_root : empty_root) && type <= D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS) {
                    // The native PSO validates the DXIL stages against the empty root.
                    supported = mesh ? !is_vertex && readable_shader_container(*code)
                        : desktop_indirect_graphics_shader(*code, is_vertex ? D3D12_SHVER_VERTEX_SHADER : D3D12_SHVER_PIXEL_SHADER);
                }
                if (!supported) {
                    std::snprintf(why, why_size, "stage %u has no translation or supported desktop binding", type);
                    ok = false;
                    break;
                }
                if (is_vertex) vertex = true;
                else pixel = true;
            }
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS || type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS) {
            const auto* code = reinterpret_cast<const D3D12_SHADER_BYTECODE*>(stream + offset + 8);
            if (!native_mesh_root || !readable_shader_container(*code)) {
                std::snprintf(why, why_size, "mesh stage needs desktop bytecode and an empty root signature");
                ok = false;
                break;
            }
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS) {
            const auto* code = reinterpret_cast<const D3D12_SHADER_BYTECODE*>(stream + offset + 8);
            compute = code->pShaderBytecode && code->BytecodeLength;
        } else if (mesh && type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT) {
            const auto* output = reinterpret_cast<const D3D12_STREAM_OUTPUT_DESC*>(stream + offset + 8);
            if (output->NumEntries || output->NumStrides) {
                std::snprintf(why, why_size, "mesh streams cannot use stream output");
                ok = false;
                break;
            }
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT) {
            input_layout = reinterpret_cast<const D3D12_INPUT_LAYOUT_DESC*>(stream + offset + 8)->NumElements != 0;
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY) {
            topology = *reinterpret_cast<const D3D12_PRIMITIVE_TOPOLOGY_TYPE*>(stream + offset + 4);
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS) {
            auto* formats = reinterpret_cast<D3D12_RT_FORMAT_ARRAY*>(stream + offset + 4);
            for (auto& format : formats->RTFormats) format = desktop_format(format);
            targets = *formats;
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT) {
            auto* format = reinterpret_cast<DXGI_FORMAT*>(stream + offset + 4);
            *format = desktop_format(*format);
            depth = *format;
        } else if (cached_mesh && (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL ||
                                  type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1)) {
            D3D12_DEPTH_STENCIL_DESC1 state{};
            const auto size = type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL ?
                sizeof(D3D12_DEPTH_STENCIL_DESC) : sizeof(state);
            std::memcpy(&state, stream + offset + sizeof(UINT), size);
            depth_enabled = state.DepthEnable;
            depth_writes = depth_enabled && state.DepthWriteMask != D3D12_DEPTH_WRITE_MASK_ZERO;
            untracked_depth_state = state.StencilEnable || state.DepthBoundsTestEnable;
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC) {
            samples = *reinterpret_cast<const DXGI_SAMPLE_DESC*>(stream + offset + 4);
        }
        offset += desktop_stream_size(type);
    }
    if (ok && cached_mesh && (untracked_depth_state || targets.NumRenderTargets > 8 ||
        (targets.NumRenderTargets ? !pixel : !depth_writes) ||
        (depth_enabled && depth == DXGI_FORMAT_UNKNOWN) ||
        std::any_of(targets.RTFormats, targets.RTFormats + std::min<UINT>(targets.NumRenderTargets, 8),
            [](DXGI_FORMAT value) { return value == DXGI_FORMAT_UNKNOWN; }))) {
        std::snprintf(why, why_size, "cached mesh needs tracked color attachments or writable depth-only state");
        ok = false;
    }
    if (ok && (vertex || pixel || mesh || amplification) &&
        ((!vertex && !mesh) || (amplification && !mesh) || (!pixel && !cached_mesh) || compute || input_layout ||
         topology != D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE ||
         (!cached_mesh && (targets.NumRenderTargets != 1 || targets.RTFormats[0] == DXGI_FORMAT_UNKNOWN ||
                           depth != DXGI_FORMAT_UNKNOWN)) ||
         samples.Count != 1 || samples.Quality)) {
        std::snprintf(why, why_size, "desktop indirect graphics stream needs VS/PS or MS/PS, triangles, one single-sample target and no vertex layout or depth");
        ok = false;
    }
    ID3D12PipelineState* pipeline = nullptr;
    if (ok) {
        ID3D12Device2* device2 = nullptr;
        HRESULT hr = device_of(the_device)->QueryInterface(IID_PPV_ARGS(&device2));
        if (SUCCEEDED(hr)) {
            const D3D12_PIPELINE_STATE_STREAM_DESC desktop{bytes, stream};
            hr = device2->CreatePipelineState(&desktop, IID_PPV_ARGS(&pipeline));
            device2->Release();
        }
        if (FAILED(hr)) std::snprintf(why, why_size, "CreatePipelineState failed 0x%08lx", static_cast<unsigned long>(hr));
        if (FAILED(hr)) {
            wchar_t dir[MAX_PATH], path[MAX_PATH];
            if (shader_dump_dir(dir, MAX_PATH)) {
                swprintf(path, MAX_PATH, L"%s\\converted_fail.bin", dir);
                write_file(path, stream, bytes);
            }
        }
        check_device("Device CreatePipelineState", 86);
    }
    if (pipeline && (vertex || mesh) && (pixel || cached_mesh)) {
        graphics = {cached_mesh ? cached_root : static_cast<ID3D12RootSignature*>(root->own().real),
            targets.RTFormats[0], mesh, cached_mesh};
        if (cached_mesh) {
            graphics.cached_target_count = targets.NumRenderTargets;
            std::copy_n(targets.RTFormats, targets.NumRenderTargets, graphics.cached_targets);
            graphics.cached_depth = depth;
            graphics.cached_depth_writes = depth_writes;
            graphics.guest_root_params = root_info(root);
        }
    }
    return pipeline;
}

// Transient data the title addresses by GPU address (fetch tables, vertices, constants) is copied
// from its memory into a desktop upload buffer; the copy's GPU address is bound instead.
// LOCAL memcut: the buffer is a chunk of the upload ring (d3d12_upload_ring.inl), returned with the offset.
#include "startup_stubs/d3d12_upload_ring.inl"
bool ring_copy(u64 title_address, std::size_t bytes, u64* offset, ID3D12Resource** buffer)
{
    auto* dest = reserve_upload_ring(bytes, offset, buffer, nullptr);
    if (!dest) return false;
    std::memcpy(dest, reinterpret_cast<const void*>(title_address), bytes);
    return true;
}
D3D12_GPU_VIRTUAL_ADDRESS upload_from_title(u64 title_address, std::size_t bytes)
{
    u64 offset = 0;
    ID3D12Resource* buffer = nullptr;
    return ring_copy(title_address, bytes, &offset, &buffer) ? buffer->GetGPUVirtualAddress() + offset : 0;
}

// Number of contiguous readable bytes at address, up to wanted.
std::size_t readable_window(u64 address, std::size_t wanted)
{
    Timed timed(time_cx_readable);
    return x2w_local::probe_readable(static_cast<std::uintptr_t>(address), wanted);
}

// The opt-in slot 62 trace records setter arguments without forwarding the calls.
struct ComputeBindingTrace {
    struct Parameter {
        UINT setter{};           // The most recent non-constant setter for this parameter.
        u64 argument{};          // Address or handle. It is classified at dispatch, never printed.
        u64 constants{};         // Dwords covered by constant setters, within the first 64.
    };
    const Wrapper* root{};
    std::array<Parameter, 16> parameters{};
};

// Per-list state while recording.
struct Pass {
    struct CopyRange { ID3D12Resource* resource; u64 offset, bytes; };
    std::vector<CopyRange> copied_gpu_ranges; // exact CopyBufferRegion destinations on this recording
    bool pipeline_real = false;      // the bound pipeline has a desktop pipeline
    bool compute_pipeline = false;
    ID3D12RootSignature* compute_root_desktop = nullptr; // upstream: desktop compute root set by its slot 33 (local: compute_root below)
    x2w_startup::ComPtr<ID3D12RootSignature> desktop_compute_root; // upstream: last root its slot 33 set on this recording
    std::array<D3D12_GPU_VIRTUAL_ADDRESS, 64> desktop_compute_upload_srvs{};  // upstream slot 43 (X2W_UPSTREAM_COMPUTE_ROOT)
    const std::vector<RootParameter>* compute_params = nullptr;
    std::unique_ptr<ComputeBindingTrace> compute_trace;
    IndirectGraphicsPipeline indirect_graphics;
    ID3D12RootSignature* graphics_root = nullptr;
    bool targets_bound = false;
    bool targets_refused = false;  // reset/explicitly empty OM is valid for UAV-only graphics
    bool topology_ok = false;
    unsigned srv_bound = 0;          // bit per root parameter with a desktop address bound
    unsigned srv_pending = 0;        // LOCAL stall: X2W_ROOT_SRV_LAZY, root SRVs set but not copied yet
    const std::vector<RootParameter>* root = nullptr;
    // Heap-indexing flags of the title root signature last set by slot 34. While nonzero, that root signature is
    // deferred: it is not set on the desktop list, root SRVs are not forwarded, and real draws go through decide_real_root.
    UINT root_heap_flags = 0;
    ID3D12RootSignature* deferred_root = nullptr;
    x2w_gpu::ListStates states;
    // What the title bound, kept for draws through translated pipelines:
    Wrapper* pipeline = nullptr;
    Wrapper* requested_pipeline = nullptr;  // last pipeline slot 61 asked a translation for; repeats skip draw_lock
    Wrapper* requested_mesh_pipeline = nullptr;
    const std::vector<RootParameter>* requested_mesh_root = nullptr;
    UINT topology = 0;
    x2w_startup::ViewTable view_tables[16]{};
    x2w_startup::ComputeTables compute_tables;  // slot 35, read by a Dispatch handler
    u64 source_view_heap = 0;
    u64 root_address[16] = {};       // title address of each root descriptor
    std::uint32_t constants[16][8] = {};
    unsigned char constant_words_bound[16] = {};
    D3D12_VERTEX_BUFFER_VIEW vb[8] = {};
    PlacedVertexView placed_vb[8] = {};
    D3D12_INDEX_BUFFER_VIEW ib = {};
    DXGI_FORMAT target_formats[8] = {};
    DXGI_FORMAT dsv_format = DXGI_FORMAT_UNKNOWN;
    bool read_depth_plane = false, read_stencil_plane = false;
    bool dsv_read_only = false;          // LOCAL (renderstate): the bound depth target is a read-only view (d3d12_draw.inl drops depth writes)
    ID3D12Resource* dsv_resource = nullptr;  // LOCAL (renderstate): desktop resource of the bound depth target
    bool targets_single_sample = false;
    bool depth_view_writable = false;
    UINT stencil_reference = 0;  // last guest OMSetStencilRef on this recording
    UINT target_count = 0;
    ID3D12Resource* target0 = nullptr;  // desktop resource of render target 0 (for per-draw dumps)
    // LOCAL meshrt: every bound target's desktop resource and the first viewport / scissor the list set, for the
    // traced-frame mesh lines (d3d12_mesh.inl mesh_trace_call). Valid for i < target_count.
    ID3D12Resource* targets[8] = {};
    D3D12_VIEWPORT viewport0 = {};
    D3D12_RECT scissor0 = {};
    ID3D12Resource* trace_targets[8]{};
    ID3D12Resource* trace_depth = nullptr;
    bool heap_set = false;
    x2w_startup::SamplerSource title_samplers;
    std::unique_ptr<x2w_startup::SamplerSnapshot> sampler_snapshot;
    x2w_startup::ComPtr<ID3D12DescriptorHeap> sampler_heap;
    ID3D12DescriptorHeap* bound_sampler_heap = nullptr;
    // LOCAL: compute root state (kind 1 constants, 2 CBV, 3 SRV, 4 UAV, 5 table)
    Wrapper* compute_root = nullptr;
    int compute_kind[16] = {};
    u64 compute_address[16] = {};
    std::uint32_t compute_constants[16][16] = {};
    UINT compute_count[16] = {};
    std::uint16_t compute_set[16] = {};  // LOCAL: constant words set since the root signature changed (entry value checks)
    // Whether the title set a viewport / scissor on this list; without them a desktop list draws no pixel.
    bool viewport_set = false;
    bool scissor_set = false;
    // Size of the automatic full-target default already forwarded to the desktop list (0 = none yet). Kept apart from
    // the title flags above so a later render target 0 of another size gets a fresh default.
    UINT64 default_viewport_w = 0, default_scissor_w = 0;
    UINT default_viewport_h = 0, default_scissor_h = 0;
    // Queries begun on the desktop list by slot 28 and not yet ended (heap, type, index). The heap pointer is
    // held without a reference until Close; releasing a heap mid-recording is invalid on desktop too.
    struct OpenQuery { ID3D12QueryHeap* heap; UINT type, index; };
    std::vector<OpenQuery> open_queries;
    // Queries ended on this recording by slot 29: what a resolve may cover. A new BeginQuery on the index removes its
    // entry, and Close clears the list with the rest of the Pass.
    std::vector<OpenQuery> ended_queries;
};
LONG render_draws, render_draws_skipped, render_pipelines_real, render_pipelines_missing;
// Heap-indexed title root signatures not set on the desktop list, and real draws refused under one, by decision.
LONG heap_roots_deferred, heap_root_refusals[x2w_startup::real_root_decision_count];

ID3D12GraphicsCommandList* recording(Wrapper* self) { return self->own().list_open ? static_cast<ID3D12GraphicsCommandList*>(self->own().real) : nullptr; }
Pass& pass_of(Wrapper* self)
{
    if (!self->own().pass) self->own().pass = new Pass;
    return *static_cast<Pass*>(self->own().pass);
}

bool indirect_copy_ranges()
{
    static const bool enabled = x2w_local::env_on("X2W_TRANSLATED_INDIRECT_COPY_RANGES");
    return enabled;
}

bool list_copied_gpu_range(Wrapper* self, ID3D12Resource* resource, u64 offset, u64 bytes)
{
    if (!indirect_copy_ranges()) return false;
    const auto& ranges = pass_of(self).copied_gpu_ranges;
    return std::any_of(ranges.begin(), ranges.end(), [&](const Pass::CopyRange& range) {
        return range.resource == resource && offset >= range.offset && bytes <= range.bytes &&
            offset - range.offset <= range.bytes - bytes;
    });
}

void note_copy_gpu_range(Wrapper* self, ID3D12Resource* resource, u64 offset, u64 bytes, bool authoritative)
{
    if (!indirect_copy_ranges()) return;
    auto& ranges = pass_of(self).copied_gpu_ranges;
    ranges.erase(std::remove_if(ranges.begin(), ranges.end(), [&](const Pass::CopyRange& range) {
        if (range.resource != resource) return false;
        return range.offset <= offset ? offset - range.offset < range.bytes : range.offset - offset < bytes;
    }), ranges.end());
    if (authoritative && ranges.size() < 64) {
        try { ranges.push_back({resource, offset, bytes}); }
        catch (const std::bad_alloc&) { return; }
        static LONG reports;
        if (first_calls(reports, 16)) log("x2w-render recorded GPU buffer copy range: offset 0x%llx, %llu bytes", offset, bytes);
    }
}
// Desktop resources rest in COMMON between lists; a list moves them as it needs and back at Close.
PlaneStates current_planes(Wrapper* self, ID3D12Resource* resource, D3D12_RESOURCE_STATES fallback)
{
    if (const auto* entry = pass_of(self).states.find(resource)) return {entry->depth, entry->stencil};
    return {fallback, fallback};
}
void require_planes(Wrapper* self, ID3D12Resource* resource, D3D12_RESOURCE_STATES depth, D3D12_RESOURCE_STATES stencil)
{
    pass_of(self).states.require(static_cast<ID3D12GraphicsCommandList*>(self->own().real), resource, depth, stencil);
}
void require_state(Wrapper* self, ID3D12Resource* resource, D3D12_RESOURCE_STATES state)
{
    require_planes(self, resource, state, state);
}
void require_bound_depth(Wrapper* self, D3D12_RESOURCE_STATES shader_reads = D3D12_RESOURCE_STATE_COMMON)
{
    const auto& pass = pass_of(self);
    const auto read = D3D12_RESOURCE_STATE_DEPTH_READ | shader_reads;
    require_planes(self, pass.dsv_resource, pass.read_depth_plane ? read : D3D12_RESOURCE_STATE_DEPTH_WRITE,
                   pass.read_stencil_plane ? read : D3D12_RESOURCE_STATE_DEPTH_WRITE);
}
void require_sampled(Wrapper* self, ID3D12Resource* resource, D3D12_RESOURCE_STATES reads)
{
    const auto& pass = pass_of(self);
    if (pass.dsv_resource == resource && (pass.read_depth_plane || pass.read_stencil_plane)) require_bound_depth(self, reads);
    else require_state(self, resource, reads);
}
void use_as_target(Wrapper* self, ID3D12Resource* resource) { require_state(self, resource, D3D12_RESOURCE_STATE_RENDER_TARGET); }

#include "startup_stubs/mesh_xmem_handoff.inl"
volatile LONG trace_uav_busy_frame;
#include "startup_stubs/red_producer_diagnostic.inl"
#include "startup_stubs/cx_input_dump.inl"  // LOCAL lt-lead probe: X2W_CX_TRACE_CLEARS, X2W_CX_DUMP_INPUTS

void discard_red_producer(ID3D12GraphicsCommandList* list) { red_producers().discard(list); }
void release_targets(Wrapper* self)
{
    Pass& pass = pass_of(self);
    pass.states.close(static_cast<ID3D12GraphicsCommandList*>(self->own().real));
    pass = Pass{};
}

// A dropped recording (Reset, or its list destroyed) names no resource any more. The raw pointers the pass keeps for
// Close go first, then the holds that kept those resources alive (X2W_D3D12_RELEASE_RETIRED). Without the switch
// nothing changes.
void retire_drop_recording(Wrapper* list)
{
    if (!release_retired_enabled()) return;
    if (list->own().pass) pass_of(list).states.clear();
    retire_discard_holds(list);
}

// A reset list has no viewport or scissor and uses the default stencil reference.
void reset_list_dynamic_state(Wrapper* self)
{
    if (!self->own().pass) return;
    pass_of(self).stencil_reference = 0;
    pass_of(self).viewport_set = false;
    pass_of(self).scissor_set = false;
    pass_of(self).default_viewport_w = pass_of(self).default_scissor_w = 0;
    pass_of(self).default_viewport_h = pass_of(self).default_scissor_h = 0;
}

// List slot 20, IASetPrimitiveTopology. 0x11 is a console-only rectangle list: translated draws and indirect
// non-indexed draws expand it to triangles; the topology is not forwarded as such.
void __stdcall list_topology(Wrapper* self, UINT topology)
{
    pass_of(self).topology_ok = topology >= 1 && topology <= 5;
    pass_of(self).topology = topology;
    if (auto* list = recording(self); list && pass_of(self).topology_ok) list->IASetPrimitiveTopology(static_cast<D3D12_PRIMITIVE_TOPOLOGY>(topology));
}
// List slot 25, SetPipelineState.
void __stdcall list_pipeline(Wrapper* self, void* pipeline)
{
    Wrapper* w = as_wrapper(pipeline);
    Pass& pass = pass_of(self);
    const bool available = w && w->own().kind == pipeline_kind && w->own().real;
    pass.indirect_graphics = available ? w->own().indirect_graphics : IndirectGraphicsPipeline{};
    // Keep direct draws on their existing translation path. Slot 61 binds
    // a recognized desktop graphics pipeline just before it records the draw.
    pass.pipeline_real = available && !pass.indirect_graphics.root;
    pass.compute_pipeline = pass.pipeline_real && w->own().compute_pipeline;
    pass.pipeline = w && w->own().kind == pipeline_kind ? w : nullptr;
    InterlockedIncrement(pass.pipeline_real ? &render_pipelines_real : &render_pipelines_missing);
    if (auto* list = recording(self); list && pass.pipeline_real) list->SetPipelineState(static_cast<ID3D12PipelineState*>(w->own().real));
}
// List slot 33 (desktop 29 + 4), SetComputeRootSignature.
// Slot 26: the console's UAV records have the desktop layout, with wrapped resource pointers.
// Desktop transitions are owned by the shim, so the other barrier types are left unhandled.
UINT __stdcall list_resource_barriers(Wrapper* self, UINT count, const D3D12_RESOURCE_BARRIER* barriers)
{
    static_assert(sizeof(D3D12_RESOURCE_BARRIER) == 32);
    UINT forwarded = 0;
    auto* list = recording(self);
    const auto type = list ? list->GetType() : D3D12_COMMAND_LIST_TYPE_COPY;
    if ((type == D3D12_COMMAND_LIST_TYPE_DIRECT || type == D3D12_COMMAND_LIST_TYPE_COMPUTE) &&
        count && barriers && x2w_local::readable(barriers, std::size_t{count} * sizeof(*barriers))) {
        for (UINT i = 0; i < count; ++i) {
            D3D12_RESOURCE_BARRIER barrier;
            std::memcpy(&barrier, barriers + i, sizeof(barrier));
            if (barrier.Type != D3D12_RESOURCE_BARRIER_TYPE_UAV || barrier.Flags != D3D12_RESOURCE_BARRIER_FLAG_NONE)
                continue;
            if (barrier.UAV.pResource) {
                Wrapper* resource = as_wrapper(barrier.UAV.pResource);
                if (!resource || resource->own().kind != resource_kind || !resource->own().real) continue;
                auto* real = static_cast<ID3D12Resource*>(resource->own().real);
                Microsoft::WRL::ComPtr<ID3D12Device> owner;
                if (FAILED(real->GetDevice(IID_PPV_ARGS(&owner))) || owner.Get() != device_of(self->own().list_device)) continue;
                if (!(real->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) {
                    if (x2w_gpu::gpu_abi_on() && gpu_depth_barrier(self, resource)) ++forwarded;
                    continue;
                }
                barrier.UAV.pResource = real;
            }
            list->ResourceBarrier(1, &barrier);
            ++forwarded;
        }
    }
    static LONG reports;
    if (count && first_calls(reports, 8))
        log("x2w-d3d12 CommandList::slot26(ResourceBarrier): forwarded=%u refused=%u", forwarded, count - forwarded);
    return 0; // no console barrier actions to perform on the caller's side
}

void __stdcall list_compute_root_desktop(Wrapper* self, void* root)
{
    auto* list = recording(self);
    if (!list) return;
    Wrapper* w = as_wrapper(root);
    Pass& pass = pass_of(self);
    const bool real = w && w->own().kind == root_signature_kind && w->own().real && !root_heap_flags(w);
    pass.compute_root_desktop = real ? static_cast<ID3D12RootSignature*>(w->own().real) : nullptr;
    pass.compute_tables.set_signature(w ? root_info(w) : nullptr);  // a different signature leaves no table bound
    pass.compute_params = real ? root_info(w) : nullptr;
    if (compute_binding_trace_enabled() && (!pass.compute_trace || pass.compute_trace->root != w)) {
        pass.compute_trace = std::make_unique<ComputeBindingTrace>();
        pass.compute_trace->root = w;
    }
    if (pass.compute_root_desktop) {
        if (pass.desktop_compute_root.Get() != pass.compute_root_desktop)
            pass.desktop_compute_upload_srvs.fill(0);
        pass.desktop_compute_root = pass.compute_root_desktop;
        list->SetComputeRootSignature(pass.compute_root_desktop);
    }
}

// Query slots 28 to 30, as a hypothesis (command_list_slot_names.h): no fixture here establishes which console
// methods sit there, so they are bound only with X2W_D3D12_QUERY_SLOTS=1. The heap reaches device slot 39 with the
// desktop desc. Only occlusion queries are
// forwarded (shortcut: other heap types are refused until a caller needs them); a call that does not fit the
// heap it names is refused and logged with its values instead of reaching the desktop list.
enum QueryMethod { begin_query, end_query, resolve_query };
void refuse_query(QueryMethod method, const char* reason, UINT type, UINT index, UINT count)
{
    static const char* const names[] = {"BeginQuery", "EndQuery", "ResolveQueryData"};
    static LONG refusals[3];  // the first 16 refusals of each method are logged
    if (first_calls(refusals[method], 16))
        log("x2w-render %s refused: %s (type %u, index %u, count %u)", names[method], reason, type, index, count);
}
// The desktop heap behind a wrapped occlusion heap that holds [index, index + count), or null with the reason.
// Only a DIRECT recording is supported in this first slice.
ID3D12QueryHeap* occlusion_heap(ID3D12GraphicsCommandList* list, void* heap, UINT type, UINT index, UINT count,
                               const char** reason)
{
    if (!list) return *reason = "list is closed", nullptr;
    if (list->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT) return *reason = "requires a direct command list", nullptr;
    const Wrapper* w = as_wrapper(heap);
    if (!w || w->own().kind != query_heap_kind || !w->own().real) return *reason = "not a desktop query heap", nullptr;
    const auto& desc = w->own().query_heap;
    if (desc.Type != D3D12_QUERY_HEAP_TYPE_OCCLUSION) return *reason = "only occlusion heaps are forwarded", nullptr;
    if (type != D3D12_QUERY_TYPE_OCCLUSION && type != D3D12_QUERY_TYPE_BINARY_OCCLUSION)
        return *reason = "query type does not fit an occlusion heap", nullptr;
    if (index >= desc.Count || count > desc.Count - index) return *reason = "outside the heap's count", nullptr;
    return static_cast<ID3D12QueryHeap*>(w->own().real);
}
// List slot 28, BeginQuery(heap, type, index).
void __stdcall list_begin_query(Wrapper* self, void* heap, UINT type, UINT index)
{
    auto* list = recording(self);
    const char* reason = nullptr;
    ID3D12QueryHeap* real = occlusion_heap(list, heap, type, index, 1, &reason);
    if (!real) return refuse_query(begin_query, reason, type, index, 1);
    Pass& pass = pass_of(self);
    for (const auto& open : pass.open_queries)
        if (open.heap == real && open.index == index) return refuse_query(begin_query, "query already begun", type, index, 1);
    auto& ended = pass.ended_queries;
    ended.erase(std::remove_if(ended.begin(), ended.end(),
                               [&](const Pass::OpenQuery& q) { return q.heap == real && q.index == index; }),
                ended.end());
    pass.open_queries.push_back({real, type, index});
    list->BeginQuery(real, static_cast<D3D12_QUERY_TYPE>(type), index);
}
// List slot 29, EndQuery(heap, type, index). Only a query this list began is ended.
void __stdcall list_end_query(Wrapper* self, void* heap, UINT type, UINT index)
{
    auto* list = recording(self);
    const char* reason = nullptr;
    ID3D12QueryHeap* real = occlusion_heap(list, heap, type, index, 1, &reason);
    if (!real) return refuse_query(end_query, reason, type, index, 1);
    Pass& pass = pass_of(self);
    auto& open = pass.open_queries;
    const auto found = std::find_if(open.begin(), open.end(),
                                    [&](const Pass::OpenQuery& q) { return q.heap == real && q.type == type && q.index == index; });
    if (found == open.end()) return refuse_query(end_query, "not begun on this list", type, index, 1);
    open.erase(found);
    pass.ended_queries.push_back({real, type, index});
    list->EndQuery(real, static_cast<D3D12_QUERY_TYPE>(type), index);
}
// At Close: a query begun on the desktop list and never ended would make the desktop Close fail and drop the whole
// list, where the console call it stands for cost nothing more than that query's result.
void end_open_queries(Wrapper* self, ID3D12GraphicsCommandList* list)
{
    if (!self->own().pass) return;
    auto& open = pass_of(self).open_queries;
    for (const auto& query : open) {
        refuse_query(end_query, "missing at Close, ended there", query.type, query.index, 1);
        list->EndQuery(query.heap, static_cast<D3D12_QUERY_TYPE>(query.type), query.index);
    }
    open.clear();
}
// List slot 30, ResolveQueryData(heap, type, start, count, destination, offset). The destination is a buffer on
// a readback heap, which stays in COPY_DEST, so it is not moved through the list's COMMON state tracking.
void __stdcall list_resolve_query(Wrapper* self, void* heap, UINT type, UINT start, UINT count, void* destination, u64 offset)
{
    auto* list = recording(self);
    const char* reason = nullptr;
    ID3D12QueryHeap* real = occlusion_heap(list, heap, type, start, count, &reason);
    if (!real) return refuse_query(resolve_query, reason, type, start, count);
    // Resolving a query that was never ended, or ended with the other occlusion type, reads uninitialized data and
    // can remove the device: every index in the range must have a matching EndQuery on this recording.
    const auto& ended = pass_of(self).ended_queries;
    for (UINT i = 0; i < count; ++i)
        if (std::none_of(ended.begin(), ended.end(), [&](const Pass::OpenQuery& q) {
                return q.heap == real && q.index == start + i && q.type == type;
            }))
            return refuse_query(resolve_query, "an index in the range was not ended with this type on this recording", type,
                                start, count);
    const Wrapper* w = as_wrapper(destination);
    if (!w || w->own().kind != resource_kind || !w->own().real)
        return refuse_query(resolve_query, "destination has no desktop resource", type, start, count);
    // Map of a resource with title memory returns that memory, which a desktop resolve never reaches.
    if (w->own().title_memory)
        return refuse_query(resolve_query, "destination has title memory", type, start, count);
    auto* resource = static_cast<ID3D12Resource*>(w->own().real);
    D3D12_HEAP_PROPERTIES properties{};
    const auto desc = resource->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || FAILED(resource->GetHeapProperties(&properties, nullptr)) ||
        properties.Type != D3D12_HEAP_TYPE_READBACK)
        return refuse_query(resolve_query, "destination is not a readback buffer", type, start, count);
    const u64 bytes = u64{count} * sizeof(UINT64);
    if ((offset & 7) || offset > desc.Width || bytes > desc.Width - offset)
        return refuse_query(resolve_query, "outside the destination buffer", type, start, count);
    list->ResolveQueryData(real, static_cast<D3D12_QUERY_TYPE>(type), start, count, resource, offset);
}
// Device slot 39, CreateQueryHeap(desc, riid, ppv): forwarded, and the desc kept for the query slots.
HRESULT __stdcall device_create_query_heap(Wrapper* self, const D3D12_QUERY_HEAP_DESC* desc, const GUID* iid, void** out)
{
    const auto hr = static_cast<HRESULT>(create_slot<device_kind, 39, 1>(self, reinterpret_cast<u64>(desc),
                                                                         reinterpret_cast<u64>(iid), reinterpret_cast<u64>(out),
                                                                         0, 0, 0, 0, 0));
    if (Wrapper* w = SUCCEEDED(hr) && desc && out ? as_wrapper(*out) : nullptr; w && w->own().kind == query_heap_kind)
        w->own().query_heap = *desc;
    return hr;
}

// Slot-61 calls by outcome and signature kind (IndirectCommand; none counts placeholder and unknown
// signatures), reported with the render totals. 64-bit and never reset, so they neither wrap nor stall.
enum { indirect_forwarded, indirect_graphics_pipeline, indirect_compute_pipeline, indirect_placeholder, indirect_other,
       indirect_outcomes };
volatile LONG64 indirect_calls[indirect_outcomes][5];

struct IndirectAddressAlias {
    ID3D12Resource* resource = nullptr;
    u64 offset = 0;
};

IndirectAddressAlias indirect_address_alias(Wrapper* device, u64 address, u64 bytes);

struct IndirectIndexView {
    D3D12_INDEX_BUFFER_VIEW view{};
    ID3D12Resource* resource = nullptr;
    u64 offset = 0;
    bool upload = false;
};
const char* draw_translated_indirect(Wrapper* self, ID3D12CommandSignature* signature, ID3D12Resource* argument,
                                     u64 offset, const IndirectIndexView& indices);
u64 known_pipeline_key(Wrapper* pipeline);

namespace indirect_other_refusal_code {
constexpr const char* list_closed = "list_closed";
constexpr const char* invalid_list_type = "invalid_list_type";
constexpr const char* invalid_command_count = "invalid_command_count";
constexpr const char* count_offset_without_buffer = "count_offset_without_buffer";
constexpr const char* signature_device_mismatch = "signature_device_mismatch";
constexpr const char* graphics_multi_command = "graphics_multi_command";
constexpr const char* cached_mesh_attachments = "cached_mesh_attachments";
constexpr const char* no_render_target = "no_render_target";
constexpr const char* multiple_render_targets = "multiple_render_targets";
constexpr const char* depth_target_bound = "depth_target_bound";
constexpr const char* missing_viewport_or_scissor = "missing_viewport_or_scissor";
constexpr const char* target_format_mismatch = "target_format_mismatch";
constexpr const char* rect_list_indexed = "rect_list_indexed";
constexpr const char* unsupported_topology = "unsupported_topology";
constexpr const char* multisample_target = "multisample_target";
constexpr const char* compute_root_mismatch = "compute_root_mismatch";
constexpr const char* dispatch_root_multi_command = "dispatch_root_multi_command";
constexpr const char* missing_argument_resource = "missing_argument_resource";
constexpr const char* invalid_argument_offset = "invalid_argument_offset";
constexpr const char* argument_address_overflow = "argument_address_overflow";
constexpr const char* argument_outside_placed_buffer = "argument_outside_placed_buffer";
constexpr const char* argument_not_common_state = "argument_not_common_state";
constexpr const char* invalid_index_view = "invalid_index_view";
constexpr const char* index_view_overlap = "index_view_overlap";
constexpr const char* index_not_common_state = "index_not_common_state";
constexpr const char* index_unaligned_address = "index_unaligned_address";
constexpr const char* upload_index_unaligned = "upload_index_unaligned";
constexpr const char* index_no_desktop_range = "index_no_desktop_range";
constexpr const char* aliased_index_unaligned = "aliased_index_unaligned";
constexpr const char* mesh_xmem_read = "mesh_xmem_read";
constexpr const char* rect_live_compute_root = "rect_live_compute_root";
constexpr const char* rect_conversion_unavailable = "rect_conversion_unavailable";
constexpr const char* count_missing_resource = "count_missing_resource";
constexpr const char* count_invalid_offset = "count_invalid_offset";
constexpr const char* count_outside_placed_buffer = "count_outside_placed_buffer";
constexpr const char* count_not_common_state = "count_not_common_state";
constexpr const char* unknown = "unknown";
}  // namespace indirect_other_refusal_code

// The guest method has independent resource and byte-offset arguments.
void __stdcall list_execute_indirect_upstream(Wrapper* self, void* signature, UINT count, void* argument, u64 offset,
                                     void* count_buffer, u64 count_offset)
{
    // Every refusal is counted under the signature's kind, including those before the signature is checked.
    Wrapper* sig = as_wrapper(signature);
    IndirectCommand kind = sig && sig->own().kind == command_signature_kind ? sig->own().indirect_command : IndirectCommand::none;
    Wrapper* refusal_pipeline = nullptr;
    u64 trace_call = 0;
    const auto refuse = [&kind, &refusal_pipeline, &trace_call](const char* reason, int outcome = indirect_other,
                                                               const char* code = nullptr) {
        InterlockedIncrement64(&indirect_calls[outcome][static_cast<int>(kind)]);
        if (trace_call)
            log("x2w-mesh86-trace call=%llu stage=record outcome=refused reason=%s", trace_call, reason);
        static LONG refusals;
        if (outcome == indirect_other) {
            const u64 key = known_pipeline_key(refusal_pipeline);
            const char* reason_code = code ? code : indirect_other_refusal_code::unknown;
            const bool first_pair = indirect_other_refusal_census.note(key, reason_code);
            if (first_pair || first_calls(refusals, 16))
                log("x2w-render ExecuteIndirect refused, key %016llx [%s]: %s", key, reason_code, reason);
            return;
        }
        if (first_calls(refusals, 16)) log("x2w-render ExecuteIndirect refused: %s", reason);
    };
    auto* list = recording(self);
    if (!list) return refuse("list is closed", indirect_other, indirect_other_refusal_code::list_closed);
    if (kind == IndirectCommand::dispatch_mesh && mesh_xmem_trace_ordinal()) {
        const auto& pass = pass_of(self);
        if (pass.pipeline && pass.pipeline->own().mesh86_key) trace_call = select_mesh_xmem_trace_call();
        if (trace_call) trace_mesh_xmem_inputs(self, pass, trace_call, argument, offset);
    }
    const auto type = list->GetType();
    if (type != D3D12_COMMAND_LIST_TYPE_DIRECT && type != D3D12_COMMAND_LIST_TYPE_COMPUTE)
        return refuse("requires a direct or compute command list", indirect_other, indirect_other_refusal_code::invalid_list_type);
    // Only a desktop Dispatch signature runs more than one command or reads a count buffer here.
    constexpr UINT max_dispatch_commands = 1u << 16;
    if (!count || count > max_dispatch_commands)
        return refuse("command count is zero or above the supported maximum", indirect_other, indirect_other_refusal_code::invalid_command_count);
    if (!count_buffer && count_offset)
        return refuse("count offset without a count buffer", indirect_other, indirect_other_refusal_code::count_offset_without_buffer);
    if (!sig || sig->own().kind != command_signature_kind || !sig->own().real)
        return refuse("missing or placeholder signature", indirect_placeholder);
    const Wrapper* device = self->own().list_device;
    if (!device || sig->own().signature_device != device)
        return refuse("requires a supported signature from this device", indirect_other, indirect_other_refusal_code::signature_device_mismatch);
    const auto command = sig->own().indirect_command;
    const bool indexed = command == IndirectCommand::draw_indexed;
    const bool mesh = command == IndirectCommand::dispatch_mesh;
    const bool graphics = indexed || mesh || command == IndirectCommand::draw;
    Pass& pass = pass_of(self);
    refusal_pipeline = pass.pipeline;
    const bool translated = (command == IndirectCommand::draw || indexed) && pass.pipeline && !pass.pipeline->own().real &&
        (pass.graphics_root || pass.deferred_root) && x2w_local::env_on("X2W_TRANSLATED_INDIRECT_DRAW");
    u64 bytes = sig->own().indirect_bytes;
    if (graphics && (count != 1 || count_buffer))
        return refuse("indirect graphics requires one command and no count buffer", indirect_other, indirect_other_refusal_code::graphics_multi_command);
    if (graphics) {
        if (indexed) bytes = sizeof(D3D12_DRAW_INDEXED_ARGUMENTS);
        else if (mesh) bytes = sizeof(D3D12_DISPATCH_MESH_ARGUMENTS);
        else bytes = sizeof(D3D12_DRAW_ARGUMENTS);
        if (type != D3D12_COMMAND_LIST_TYPE_DIRECT || (!translated && (!pass.indirect_graphics.root ||
            pass.indirect_graphics.root != pass.graphics_root || pass.indirect_graphics.mesh != mesh))) {
            // A console pipeline (no desktop pipeline) used only here still gets the request a direct draw writes;
            // nothing is drawn.
            if (mesh && type == D3D12_COMMAND_LIST_TYPE_DIRECT && pass.pipeline && !pass.pipeline->own().real &&
                (pass.requested_mesh_pipeline != pass.pipeline || pass.requested_mesh_root != pass.root)) {
                pass.requested_mesh_pipeline = pass.pipeline;
                pass.requested_mesh_root = pass.root;
                request_slot57_mesh(pass.pipeline, pass.root);
            } else if (!mesh && pass.pipeline && !pass.pipeline->own().real && pass.requested_pipeline != pass.pipeline) {
                pass.requested_pipeline = pass.pipeline;
                request_translation(pass.pipeline);
            }
            if (mesh && pass.pipeline && pass.pipeline->own().mesh86_key) {
                static LONG mesh_cache_refusals;
                if (first_calls(mesh_cache_refusals, 16))
                    log("x2w-render slot86 mesh %016llx: no matching desktop pipeline/root for indirect execution",
                        pass.pipeline->own().mesh86_key);
            }
            return refuse("indirect graphics requires a direct list with a matching desktop pipeline and root",
                          indirect_graphics_pipeline);
        }
        if (pass.indirect_graphics.cached_mesh) {
            const auto& pso = pass.indirect_graphics;
            if (!pass.targets_bound || !pass.viewport_set || !pass.scissor_set || !pass.targets_single_sample ||
                pass.target_count != pso.cached_target_count || pass.dsv_format != pso.cached_depth ||
                (pso.cached_depth_writes && !pass.depth_view_writable) ||
                !std::equal(pso.cached_targets, pso.cached_targets + pso.cached_target_count, pass.target_formats))
                return refuse("cached mesh requires matching color and depth attachments, viewport and scissor",
                              indirect_other, indirect_other_refusal_code::cached_mesh_attachments);
            if (!pass.root || pass.root != pso.guest_root_params || pass.root_heap_flags)
                return refuse("cached mesh requires a recorded root without heap indexing", indirect_graphics_pipeline);
            for (std::size_t i = 0; i < pass.root->size(); ++i) {
                if (i == pso.cached_srv_root) continue;
                const auto mask = (1u << (*pass.root)[i].constants_count) - 1;
                if ((pass.constant_words_bound[i] & mask) != mask)
                    return refuse("cached mesh root constants are incomplete", indirect_graphics_pipeline);
            }
        } else if (!pass.targets_bound || pass.target_count != 1 || !pass.target0 ||
            (!translated && pass.target_formats[0] != pass.indirect_graphics.target_format) ||
            pass.dsv_format != DXGI_FORMAT_UNKNOWN || !pass.viewport_set || !pass.scissor_set) {
            const char* code =
                pass.target_count > 1                                 ? indirect_other_refusal_code::multiple_render_targets
                : (!pass.targets_bound || !pass.target_count || !pass.target0) ? indirect_other_refusal_code::no_render_target
                : pass.dsv_format != DXGI_FORMAT_UNKNOWN              ? indirect_other_refusal_code::depth_target_bound
                : (!pass.viewport_set || !pass.scissor_set)           ? indirect_other_refusal_code::missing_viewport_or_scissor
                                                                      : indirect_other_refusal_code::target_format_mismatch;
            return refuse("indirect graphics requires one matching target, viewport and scissor", indirect_other, code);
        }
        const bool rect_list = !mesh && pass.topology == 0x11;
        if (rect_list && indexed && !translated)
            return refuse("indirect rectangle list requires a non-indexed draw", indirect_other, indirect_other_refusal_code::rect_list_indexed);
        if (!mesh && !rect_list && pass.topology != D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST)
            return refuse("indirect draw requires triangle-list or rectangle-list topology", indirect_other, indirect_other_refusal_code::unsupported_topology);
        if (!pass.indirect_graphics.cached_mesh) {
            const auto samples = pass.target0->GetDesc().SampleDesc;
            if (samples.Count != 1 || samples.Quality)
                return refuse("draw requires a single-sample render target", indirect_other, indirect_other_refusal_code::multisample_target);
        }
    } else if (!pass.compute_pipeline || !pass.compute_root_desktop) {
        return refuse("requires a desktop compute pipeline and root", indirect_compute_pipeline);
    } else if (sig->own().signature_root && sig->own().signature_root != pass.compute_root_desktop) {
        return refuse("signature root constants require the bound compute root", indirect_other, indirect_other_refusal_code::compute_root_mismatch);
    } else if (sig->own().signature_root && (count != 1 || count_buffer)) {
        // Batched and count-buffer Dispatch is supported for rootless signatures only in this slice.
        return refuse("a Dispatch signature with root constants requires one command and no count buffer",
                      indirect_other, indirect_other_refusal_code::dispatch_root_multi_command);
    }
    Wrapper* buffer = as_wrapper(argument);
    if (!buffer || buffer->own().kind != resource_kind || !buffer->own().real)
        return refuse("argument has no desktop resource", indirect_other, indirect_other_refusal_code::missing_argument_resource);
    const auto base = reinterpret_cast<u64>(buffer->own().title_memory);
    if (!base || (offset & 3) || offset > ~u64{} - base)
        return refuse("invalid argument offset or unmapped buffer", indirect_other, indirect_other_refusal_code::invalid_argument_offset);
    // Every record up to the maximum must lie in the buffer: the count word can only lower the number executed.
    const u64 span = u64{count - 1} * sig->own().indirect_bytes + bytes;
    if (span < bytes || base + offset > ~u64{} - span)
        return refuse("argument records overflow the address space", indirect_other, indirect_other_refusal_code::argument_address_overflow);
    const auto found = placed_buffer_range(device, base + offset, span);
    if (found.reject || found.owner != buffer)
        return refuse("argument record is outside a unique placed buffer", indirect_other, indirect_other_refusal_code::argument_outside_placed_buffer);
    const auto* info = resource_info(buffer);
    if (!info || info->initial_state != D3D12_RESOURCE_STATE_COMMON)
        return refuse("argument buffer does not use COMMON state tracking", indirect_other, indirect_other_refusal_code::argument_not_common_state);
    auto* resource = static_cast<ID3D12Resource*>(buffer->own().real);
    IndirectIndexView index_view;
    if (indexed) {
        auto view = pass.ib;
        const UINT alignment = view.Format == DXGI_FORMAT_R16_UINT ? 2 : view.Format == DXGI_FORMAT_R32_UINT ? 4 : 0;
        if (!alignment || !view.BufferLocation || !view.SizeInBytes ||
            view.BufferLocation % alignment || view.SizeInBytes % alignment)
            return refuse("invalid index view", indirect_other, indirect_other_refusal_code::invalid_index_view);
        const auto indices = placed_buffer_range(device, view.BufferLocation, view.SizeInBytes);
        if (indices.reject)
            return refuse("index view crosses or overlaps placed buffers", indirect_other, indirect_other_refusal_code::index_view_overlap);
        if (indices.owner && indices.owner->own().real) {
            const auto* index_info = resource_info(indices.owner);
            if (!index_info || index_info->initial_state != D3D12_RESOURCE_STATE_COMMON)
                return refuse("index buffer does not use COMMON state tracking", indirect_other, indirect_other_refusal_code::index_not_common_state);
            auto* index = static_cast<ID3D12Resource*>(indices.owner->own().real);
            const auto index_base = reinterpret_cast<u64>(indices.owner->own().title_memory);
            const auto index_offset = view.BufferLocation - index_base;
            const auto native_base = index->GetGPUVirtualAddress();
            if (!native_base || index_offset > ~u64{} - native_base || (native_base + index_offset) % alignment)
                return refuse("index view has no aligned desktop address", indirect_other, indirect_other_refusal_code::index_unaligned_address);
            view.BufferLocation = native_base + index_offset;
            index_view = {view, index, index_offset};
        } else {
            const auto upload = translated && !indices.owner
                ? list_upload_range(self, view.BufferLocation, view.SizeInBytes) : x2w_upload::Allocation{};
            // The compute validator reads 32-bit words, including for R16_UINT. Do not
            // load an adjacent recording upload allocation when extracting a half-word.
            if (upload.resource) {
                if ((upload.offset | view.SizeInBytes) & 3)
                    return refuse("upload index view is not dword-aligned and dword-sized",
                                  indirect_other, indirect_other_refusal_code::upload_index_unaligned);
                view.BufferLocation = upload.gpu;
                index_view = {view, upload.resource, upload.offset, true};
            } else {
                const auto alias = translated && !indices.owner && !list_upload_range(self, view.BufferLocation, 1).resource
                    ? indirect_address_alias(self->own().list_device, view.BufferLocation, view.SizeInBytes) : IndirectAddressAlias{};
                if (!alias.resource)
                    return refuse("index view has no proven desktop buffer range", indirect_other, indirect_other_refusal_code::index_no_desktop_range);
                const auto gpu = alias.resource->GetGPUVirtualAddress();
                if (!gpu || ((alias.offset | view.SizeInBytes) & 3))
                    return refuse("aliased index view is not dword-aligned and dword-sized",
                                  indirect_other, indirect_other_refusal_code::aliased_index_unaligned);
                view.BufferLocation = gpu + alias.offset;
                index_view = {view, alias.resource, alias.offset};
            }
        }
    }
    if (translated) {
        const char* failure = draw_translated_indirect(self, static_cast<ID3D12CommandSignature*>(sig->own().real), resource,
                                                       offset, index_view);
        if (failure) return refuse(failure, indirect_graphics_pipeline);
        InterlockedIncrement64(&indirect_calls[indirect_forwarded][static_cast<int>(kind)]);
        return;
    }
    auto argument_state = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    if (index_view.resource) {
        if (index_view.resource == resource) argument_state |= D3D12_RESOURCE_STATE_INDEX_BUFFER;
        else require_state(self, index_view.resource, D3D12_RESOURCE_STATE_INDEX_BUFFER);
        list->IASetIndexBuffer(&index_view.view);
    }
    const bool rect_list = graphics && !mesh && pass.topology == 0x11;
    ID3D12Resource* scratch = nullptr;
    ID3D12Resource* draw_arguments = resource;
    u64 draw_offset = offset;
    const UINT mesh_srv = mesh && pass.indirect_graphics.cached_mesh ? pass.indirect_graphics.cached_srv_root : UINT_MAX;
    const bool leased_arguments = mesh && mesh_xmem_indirect() && info->placement == Placement::copied_buffer &&
        (!pass.indirect_graphics.cached_mesh || mesh_srv != UINT_MAX);
    MeshXMemRead inputs[2]{};
    size_t input_count = 0;
    if (mesh_srv != UINT_MAX)
        inputs[input_count++] = {pass.root_address[mesh_srv], sizeof(UINT), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
    if (leased_arguments)
        inputs[input_count++] = {base + offset, bytes, argument_state};
    if (input_count) {
        if (const char* failure = record_mesh_xmem_reads(self, inputs, input_count, trace_call))
            return refuse(failure, mesh_srv != UINT_MAX ? indirect_graphics_pipeline : indirect_other,
                          indirect_other_refusal_code::mesh_xmem_read);
        if (leased_arguments) {
            draw_arguments = inputs[input_count - 1].resource;
            draw_offset = inputs[input_count - 1].offset;
        }
    }
    if (rect_list) {
        if (std::any_of(pass.desktop_compute_upload_srvs.begin(), pass.desktop_compute_upload_srvs.end(),
                        [](u64 address) { return address != 0; }))
            return refuse("rectangle-list conversion would drop live compute root arguments",
                          indirect_other, indirect_other_refusal_code::rect_live_compute_root);
        // The producer's count names three vertices per rectangle; the desktop draw needs six. Convert on the GPU
        // into a scratch record and draw from that. The compute root signature and bindings of the list are replaced.
        // The four words are copied into the scratch first (the pass never reads the title's buffer), so the source only
        // needs COPY_SOURCE next to its argument state.
        require_state(self, resource, argument_state | D3D12_RESOURCE_STATE_COPY_SOURCE);
        scratch = recording_rect_scratch(self);
        if (!scratch || !x2w_startup::rect_indirect_arguments().convert(list, resource, offset, scratch))
            return refuse("rectangle-list argument conversion is unavailable",
                          indirect_other, indirect_other_refusal_code::rect_conversion_unavailable);
        if (pass.desktop_compute_root) list->SetComputeRootSignature(pass.desktop_compute_root.Get());
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        draw_arguments = scratch;
        draw_offset = 0;
    }
    if (graphics) {
        list->SetPipelineState(static_cast<ID3D12PipelineState*>(pass.pipeline->own().real));
        list->SetGraphicsRootSignature(pass.graphics_root);
        if (pass.indirect_graphics.cached_mesh) {
            const UINT srv = pass.indirect_graphics.cached_srv_root;
            if (srv != UINT_MAX)
                list->SetGraphicsRootShaderResourceView(srv, inputs[0].resource->GetGPUVirtualAddress() + inputs[0].offset);
            for (UINT i = 0; i < pass.root->size(); ++i) {
                if (i == srv) continue;
                list->SetGraphicsRoot32BitConstants(i, (*pass.root)[i].constants_count, pass.constants[i], 0);
            }
        }
    }
    ID3D12Resource* counts = nullptr;
    if (count_buffer) {
        // The count word is four bytes in a unique placed buffer that also tracks state from COMMON.
        Wrapper* count_wrapper = as_wrapper(count_buffer);
        if (!count_wrapper || count_wrapper->own().kind != resource_kind || !count_wrapper->own().real)
            return refuse("count buffer has no desktop resource", indirect_other, indirect_other_refusal_code::count_missing_resource);
        const auto count_base = reinterpret_cast<u64>(count_wrapper->own().title_memory);
        if (!count_base || (count_offset & 3) || count_offset > ~u64{} - count_base - sizeof(UINT))
            return refuse("invalid count offset or unmapped count buffer", indirect_other, indirect_other_refusal_code::count_invalid_offset);
        const auto count_found = placed_buffer_range(device, count_base + count_offset, sizeof(UINT));
        if (count_found.reject || count_found.owner != count_wrapper)
            return refuse("count word is outside a unique placed buffer", indirect_other, indirect_other_refusal_code::count_outside_placed_buffer);
        const auto* count_info = resource_info(count_wrapper);
        if (!count_info || count_info->initial_state != D3D12_RESOURCE_STATE_COMMON)
            return refuse("count buffer does not use COMMON state tracking", indirect_other, indirect_other_refusal_code::count_not_common_state);
        counts = static_cast<ID3D12Resource*>(count_wrapper->own().real);
        if (counts != resource) require_state(self, counts, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    }
    if (!leased_arguments && !rect_list) {
        require_state(self, resource, argument_state);
    }
    const auto red_samples = graphics ? red_producer_draw_before(self, list, "indirect-draw", 0) : RedDrawSamples{};
    list->ExecuteIndirect(static_cast<ID3D12CommandSignature*>(sig->own().real), count, draw_arguments, draw_offset, counts,
                          counts ? count_offset : 0);
    red_producer_draw_after(self, red_samples);
    if (trace_call) log("x2w-mesh86-trace call=%llu stage=record outcome=recorded", trace_call);
    if (scratch) x2w_startup::RectIndirectArguments::finish(list, scratch);
    InterlockedIncrement64(&indirect_calls[indirect_forwarded][static_cast<int>(kind)]);
}
// List slot 34 (desktop 30 + 4), SetGraphicsRootSignature. The title's parameters are tracked for translated
// draws either way. A root signature that indexes a descriptor heap directly is not set on the desktop list: the
// title's heaps are never forwarded there (slot 32), so the debug layer would reject it (message ID 1419).
void __stdcall list_graphics_root(Wrapper* self, void* root)
{
    Wrapper* w = as_wrapper(root);
    Pass& pass = pass_of(self);
    const auto* params = w ? root_info(w) : nullptr;
    // A different root signature invalidates the title's cached root arguments.
    if (pass.root != params) {
        for (auto& table : pass.view_tables) table = {};
        std::fill_n(pass.root_address, 16, 0);
        std::memset(pass.constants, 0, sizeof(pass.constants));
        std::memset(pass.constant_words_bound, 0, sizeof(pass.constant_words_bound));
    }
    pass.root = params;
    pass.srv_bound = 0;
    pass.srv_pending = 0;
    // LOCAL: a root signature with heap-indexing flags is set on the desktop list as before. Upstream defers it
    // (not set; real draws on it refused) only with X2W_DEFER_HEAP_ROOTS=1.
    static const bool defer_heap_roots = x2w_local::env_on("X2W_DEFER_HEAP_ROOTS");
    const bool real = w && w->own().kind == root_signature_kind && w->own().real;
    pass.root_heap_flags = real && defer_heap_roots ? root_heap_flags(w) : 0;
    pass.deferred_root = pass.root_heap_flags ? static_cast<ID3D12RootSignature*>(w->own().real) : nullptr;
    pass.graphics_root = nullptr;
    auto* list = recording(self);
    if (!list || !real) return;
    if (!pass.root_heap_flags) {
        pass.graphics_root = static_cast<ID3D12RootSignature*>(w->own().real);
        list->SetGraphicsRootSignature(pass.graphics_root);
        return;
    }
    const LONG deferred = InterlockedIncrement(&heap_roots_deferred);
    if (deferred <= 3)
        log("x2w-render root signature %p with heap flags 0x%x not set on the desktop list (title heaps are not forwarded)",
            static_cast<void*>(w), pass.root_heap_flags);
}
DXGI_FORMAT view_format_of(u64 desktop_handle, ID3D12Resource* target)
{
    DXGI_FORMAT format = target->GetDesc().Format;
    if (const std::uint32_t id = shadow_id(desktop_handle)) {
        AcquireSRWLockShared(&note_lock);
        DXGI_FORMAT view_format = DXGI_FORMAT_UNKNOWN;
        if (id - 1 < view_records().size() && view_records()[id - 1].desc_bytes >= 4)
            std::memcpy(&view_format, view_records()[id - 1].desc, 4);
        ReleaseSRWLockShared(&note_lock);
        if (view_format != DXGI_FORMAT_UNKNOWN) format = desktop_format(view_format);
    }
    return format;
}
// The view a descriptor holds now, read from its shadow record so copies are followed. A record of the
// other kind is wrong_kind; a handle with no record falls back to the creation-time map (render targets only).
struct ResolvedView {
    ID3D12Resource* resource = nullptr;
    UINT dsv_flags = 0;
    bool wrong_kind = false;
    bool skipped = false;
};
ResolvedView resolve_view(u64 handle, int slot)
{
    ResolvedView out;
    ViewRecord record{};
    bool known = false;
    bool materialized = false;
    if (const std::uint32_t id = shadow_id(handle)) {
        AcquireSRWLockShared(&note_lock);
        known = id - 1 < view_records().size();
        if (known) {
            record = view_records()[id - 1];
            materialized = id - 1 < view_materialized().size() && view_materialized()[id - 1] != 0;
        }
        ReleaseSRWLockShared(&note_lock);
    }
    if (!known) {
        if (slot == 20 || slot == 21) out.resource = view_target(handle, slot);
        return out;
    }
    if (static_cast<int>(record.slot) != slot) {
        out.wrong_kind = true;
        return out;
    }
    if (!materialized) {
        out.skipped = true;
        return out;
    }
    out.resource = record.resource ? reinterpret_cast<ID3D12Resource*>(unwrap(record.resource)) : nullptr;
    if (slot == 21 && record.desc_bytes >= 12) std::memcpy(&out.dsv_flags, record.desc + 8, 4);
    return out;
}
bool format_has_stencil(DXGI_FORMAT format)
{
    return format == DXGI_FORMAT_D24_UNORM_S8_UINT || format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT ||
           format == DXGI_FORMAT_R24G8_TYPELESS || format == DXGI_FORMAT_R32G8X24_TYPELESS;
}
// List slot 51 (desktop 47 + 4), ClearDepthStencilView(view, flags, depth, stencil, rect count, rects). Desktop
// resources rest in COMMON here, so the target is moved to DEPTH_WRITE around the clear. The view is
// read from its shadow record, as OMSetRenderTargets does, so a descriptor the title copied clears the
// resource it now holds; a handle with no record falls back to the creation-time map.
void __stdcall list_clear_depth_stencil(Wrapper* self, u64 handle, D3D12_CLEAR_FLAGS flags, float depth, UINT8 stencil,
                                       UINT rect_count, const D3D12_RECT* rects)
{
    if (!self || !self->own().list_open) return;
    Timed timed(time_list);
    InterlockedIncrement(&draw_calls_seen);
    InterlockedIncrement64(&list_slot_calls[51]);
    InterlockedIncrement64(&list_slot_census[51]);
    if (trace_frame_active) {
        const auto ret = reinterpret_cast<u64>(_ReturnAddress()) - reinterpret_cast<u64>(GetModuleHandleW(nullptr));
        log("x2w-frame tid=%lu list=%llx slot51(0x%llx, 0x%x, %.3f, %u | %u, %p) ret=+0x%llx",
            GetCurrentThreadId(), reinterpret_cast<u64>(self), handle, static_cast<unsigned>(flags),
            depth, static_cast<unsigned>(stencil), rect_count, static_cast<const void*>(rects), ret);
    }
    const u64 desktop = desktop_cpu_handle(handle);
    const ResolvedView view = resolve_view(desktop, 21);
    ID3D12Resource* target = view.resource;
    static LONG logged;
    if (first_calls(logged, 4))
        log("x2w-render ClearDepthStencilView view=0x%llx target=%p flags=0x%x depth=%.3f stencil=%u rects=%u", handle,
            static_cast<void*>(target), static_cast<unsigned>(flags), depth, static_cast<unsigned>(stencil), rect_count);
    if (!target) {
        InterlockedIncrement(&render_target_unresolved);
        static LONG refused;
        if (first_calls(refused, 6))
            log("x2w-d3d12 ClearDepthStencilView skipped: view 0x%llx is %s", handle,
                view.wrong_kind ? "not a depth stencil view"
                                : view.skipped ? "a skipped depth stencil view" : "not a known depth stencil view");
        return;
    }
    if (!(flags & (D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL))) {
        static LONG refused;
        if (first_calls(refused, 6))
            log("x2w-d3d12 ClearDepthStencilView skipped: zero clear flags 0x%x", static_cast<unsigned>(flags));
        return;
    }
    if ((flags & D3D12_CLEAR_FLAG_DEPTH) && (view.dsv_flags & D3D12_DSV_FLAG_READ_ONLY_DEPTH)) {
        static LONG refused;
        if (first_calls(refused, 6))
            log("x2w-d3d12 ClearDepthStencilView skipped: depth clear requested on read-only depth view 0x%llx", handle);
        return;
    }
    if ((flags & D3D12_CLEAR_FLAG_STENCIL) && (view.dsv_flags & D3D12_DSV_FLAG_READ_ONLY_STENCIL)) {
        static LONG refused;
        if (first_calls(refused, 6))
            log("x2w-d3d12 ClearDepthStencilView skipped: stencil clear requested on read-only stencil view 0x%llx", handle);
        return;
    }
    const DXGI_FORMAT format = view_format_of(desktop, target);
    if ((flags & D3D12_CLEAR_FLAG_STENCIL) && !format_has_stencil(format)) {
        flags = static_cast<D3D12_CLEAR_FLAGS>(flags & ~D3D12_CLEAR_FLAG_STENCIL);
        if (!flags) return;
    }
    if (rect_count && !x2w_local::readable(rects, std::size_t{rect_count} * sizeof(D3D12_RECT))) {
        static LONG refused;
        if (first_calls(refused, 6))
            log("x2w-d3d12 ClearDepthStencilView skipped: unreadable rects array (%u rects at %p)", rect_count, static_cast<const void*>(rects));
        return;
    }
    auto* list = static_cast<ID3D12GraphicsCommandList*>(self->own().real);
    const auto before = current_planes(self, target, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    require_state(self, target, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    cx_dump::note_clear("dsv", handle, target, 0, nullptr, false, depth, stencil, static_cast<unsigned>(flags), rect_count, nullptr);
    const auto red_sample = red_producers().before(self, list, target, "clear-dsv", 0, 0);
    list->ClearDepthStencilView(D3D12_CPU_DESCRIPTOR_HANDLE{static_cast<SIZE_T>(desktop)}, flags, depth, stencil,
                                rect_count, rect_count ? rects : nullptr);
    red_producers().after(self, red_sample);
    if (pass_of(self).dsv_resource == target) require_planes(self, target, before.depth, before.stencil);
    InterlockedIncrement(&render_clears);
}
// List slot 52 (desktop 48 + 4), ClearRenderTargetView(view, colour, rect count, rects). Desktop
// resources rest in COMMON here, so the target is moved to RENDER_TARGET around the clear.
LONG clears_dropped[2];  // slot 52 dropped: [0] list not open or no colour, [1] view resolves to no target
void __stdcall list_clear_target(Wrapper* self, u64 handle, const float* colour, UINT rect_count, const D3D12_RECT* rects)
{
    if (!self->own().list_open || !colour) {
        cx_dump::note_clear("rtv", handle, nullptr, 0, colour, true, 0, 0, 0, rect_count, !colour ? "no colour" : "list not open");
        return (void)InterlockedIncrement(&clears_dropped[0]);
    }
    const u64 desktop = desktop_cpu_handle(handle);
    ID3D12Resource* target = view_target(desktop);
    cx_dump::note_clear("rtv", handle, target, 0, colour, true, 0, 0, 0, rect_count, target ? nullptr : "view resolves to no target");
    static LONG logged;
    if (first_calls(logged, 4))
        log("x2w-render ClearRenderTargetView view=0x%llx target=%p colour=%.3f %.3f %.3f %.3f rects=%u", handle,
            static_cast<void*>(target), colour[0], colour[1], colour[2], colour[3], rect_count);
    if (!target) return (void)InterlockedIncrement(&clears_dropped[1]);
    auto* list = static_cast<ID3D12GraphicsCommandList*>(self->own().real);
    use_as_target(self, target);
    const auto red_sample = red_producers().before(self, list, target, "clear-rtv", 0, 0);
    list->ClearRenderTargetView(D3D12_CPU_DESCRIPTOR_HANDLE{static_cast<SIZE_T>(desktop)}, colour, rect_count,
                                rect_count ? rects : nullptr);
    red_producers().after(self, red_sample);
    InterlockedIncrement(&render_clears);
}

// Outcome of every slot 50 call (exactly one each), reported and reset every 300 presented frames.
enum TargetBind {
    bind_bound, bind_single_range_bound, bind_not_recording, bind_count_over_8, bind_handles_unreadable,
    bind_range_past_heap, bind_wrong_kind, bind_unresolved, bind_empty, bind_depth_wrong_kind, bind_depth_unknown,
    bind_outcomes
};
LONG bound_mixed_depth, target_binds[bind_outcomes], target_unresolved_at[8];
LONG target_depth_unreadable, target_depth_unmapped;  // bound without the depth the title passed
LONG draws_without_targets;  // translated draws refused because no bind succeeded on the list (d3d12_draw.inl)
LONG state_dropped[3];  // while recording: viewports, scissors (count 0, over 16 or unreadable), blend factor unreadable
// The desktop shader-visible CBV/SRV/UAV heap that owns the GPU handle's descriptor, or null.
ID3D12DescriptorHeap* owning_view_heap(u64 gpu)
{
    u64 owner = 0;
    AcquireSRWLockShared(&note_lock);
    for (const HeapInfo& heap : heap_infos()) {
        if (heap.type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV || !heap.desktop_gpu || gpu < heap.desktop_gpu) continue;
        const u64 offset = gpu - heap.desktop_gpu;
        if (heap.increment && offset < u64{heap.count} * heap.increment && offset % heap.increment == 0) {
            owner = heap.wrapper;
            break;
        }
    }
    ReleaseSRWLockShared(&note_lock);
    return owner ? reinterpret_cast<ID3D12DescriptorHeap*>(unwrap(owner)) : nullptr;
}
// require_state assumes a resource starts in COMMON; a placed buffer made with another initial state does not.
bool tracked_from_common(u64 title_resource)
{
    const ResourceInfo* info = resource_info(as_wrapper(reinterpret_cast<const void*>(title_resource)));
    return !info || info->initial_state == D3D12_RESOURCE_STATE_COMMON;
}
// List slots 53 and 54 (desktop 49/50 + 4), ClearUnorderedAccessViewUint/Float(view GPU handle, view CPU
// handle, resource, values, rect count, rects). The CPU handle is read from its shadow record, like the
// render-target clear, so a copied descriptor clears the resource it holds now. A GPU handle that is on no
// known heap (its owning heap is bound for the clear and the translated-heap cache is reset), a resource not
// tracked from COMMON, a CPU handle that holds no unordered-access view, or a resource without unordered access is
// refused (logged, counted) instead of reaching the desktop list. The target moves to UNORDERED_ACCESS
// around the clear.
template <bool Float>
void __stdcall list_clear_uav(Wrapper* self, u64 gpu_handle, u64 cpu_handle, u64 title_resource, const void* values,
                              UINT rect_count, const D3D12_RECT* rects)
{
    if (!self->own().list_open) return;
    static LONG logged, refused;
    const char* name = Float ? "ClearUnorderedAccessViewFloat" : "ClearUnorderedAccessViewUint";
    const u64 desktop = desktop_cpu_handle(cpu_handle);
    const ResolvedView view = desktop ? resolve_view(desktop, 19) : ResolvedView{};
    ID3D12Resource* target = view.resource;
    ID3D12DescriptorHeap* view_heap = nullptr;
    const char* why = nullptr;
    if (!x2w_local::readable(values, 4 * sizeof(UINT))) why = "no clear values";
    else if (rect_count && !x2w_local::readable(rects, std::size_t{rect_count} * sizeof(D3D12_RECT))) why = "unreadable rects";
    else if (!target) why = view.wrong_kind ? "view is not an unordered-access view"
                            : view.skipped ? "view was skipped at creation" : "view holds no known resource";
    else if (!(view_heap = owning_view_heap(gpu_handle))) why = "GPU handle is on no known descriptor heap";
    else if (!(target->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) why = "resource lacks unordered access";
    else if (!tracked_from_common(title_resource)) why = "resource does not use COMMON state tracking";
    cx_dump::note_clear(Float ? "uav-float" : "uav-uint", cpu_handle, target, title_resource, values, Float, 0, 0, 0, rect_count, why);
    if (why) {
        InterlockedIncrement(&render_target_unresolved);
        if (first_calls(refused, 6)) log("x2w-d3d12 %s skipped: %s (view 0x%llx)", name, why, cpu_handle);
        return;
    }
    if (first_calls(logged, 4)) log("x2w-render %s view=0x%llx target=%p rects=%u", name, cpu_handle, static_cast<void*>(target), rect_count);
    auto* list = static_cast<ID3D12GraphicsCommandList*>(self->own().real);
    require_state(self, target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    // The clear needs the GPU descriptor's own heap bound; the next translated draw binds its heaps again.
    ID3D12DescriptorHeap* bound[] = {view_heap};
    list->SetDescriptorHeaps(1, bound);
    pass_of(self).heap_set = false;
    const D3D12_GPU_DESCRIPTOR_HANDLE gpu{gpu_handle};
    const D3D12_CPU_DESCRIPTOR_HANDLE cpu{static_cast<SIZE_T>(desktop)};
    if (Float) list->ClearUnorderedAccessViewFloat(gpu, cpu, target, static_cast<const float*>(values), rect_count, rect_count ? rects : nullptr);
    else list->ClearUnorderedAccessViewUint(gpu, cpu, target, static_cast<const UINT*>(values), rect_count, rect_count ? rects : nullptr);
    InterlockedIncrement(&render_clears);
}
// List slot 55 (desktop 51 + 4), DiscardResource(resource, region). Discard is only a hint, so a resource
// that is not a render target, depth target or unordered-access resource (needs a state the list does not
// track), an unreadable region or a region with unreadable rects is dropped; nothing reaches the desktop list.
void __stdcall list_discard(Wrapper* self, u64 title_resource, const D3D12_DISCARD_REGION* region)
{
    auto* list = recording(self);
    if (!list || !title_resource) return;
    auto* resource = reinterpret_cast<ID3D12Resource*>(unwrap(title_resource));
    if (!resource) return;
    D3D12_DISCARD_REGION safe{};
    if (region) {
        if (!x2w_local::readable(region, sizeof(*region))) return;
        safe = *region;
        if (safe.NumRects && !x2w_local::readable(safe.pRects, std::size_t{safe.NumRects} * sizeof(D3D12_RECT))) return;
    }
    if (!tracked_from_common(title_resource)) return;
    const D3D12_RESOURCE_FLAGS flags = resource->GetDesc().Flags;
    D3D12_RESOURCE_STATES state;
    if (flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) state = D3D12_RESOURCE_STATE_RENDER_TARGET;
    else if (flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) state = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    else if (flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    else return;
    require_state(self, resource, state);
    list->DiscardResource(resource, region ? &safe : nullptr);
    static LONG logged;
    if (first_calls(logged, 4)) log("x2w-render DiscardResource resource=%p rects=%u", static_cast<void*>(resource), region ? safe.NumRects : 0);
}
// List slot 50 (desktop 46 + 4), OMSetRenderTargets(count, handles, single range, depth).
// With single_range the title passes only the first handle; target i is at handle + i * the title's RTV
// stride (title_descriptor_stride: for RTV heaps the same desktop increment the title reads from the
// forwarded GetDescriptorHandleIncrementSize), and each is mapped to its desktop descriptor on its own.
// A descriptor of the wrong kind, a colour handle that holds no view,
// or a depth view that resolves to nothing refuses the whole bind:
// desktop targets are unbound and subsequent graphics work is skipped until the next valid bind.
void __stdcall list_targets(Wrapper* self, UINT count, const D3D12_CPU_DESCRIPTOR_HANDLE* handles, BOOL single_range,
                            const D3D12_CPU_DESCRIPTOR_HANDLE* depth)
{
    auto* list = recording(self);
    Pass& pass = pass_of(self);
    pass.targets_bound = false;
    pass.targets_single_sample = false;
    pass.depth_view_writable = false;
    pass.target_count = 0;
    pass.target0 = nullptr;
    pass.dsv_format = DXGI_FORMAT_UNKNOWN;
    pass.read_depth_plane = pass.read_stencil_plane = false;
    pass.dsv_read_only = false;
    pass.dsv_resource = nullptr;
    std::fill_n(pass.trace_targets, 8, nullptr);
    pass.trace_depth = nullptr;
    static LONG refused, unresolved_logged;
    const auto refuse = [&](TargetBind why) {
        pass.targets_refused = true;
        if (list) list->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
        InterlockedIncrement(&target_binds[why]);
    };
    if (!list) return refuse(bind_not_recording);
    if (count > 8) {
        if (first_calls(refused, 6)) log("x2w-d3d12 OMSetRenderTargets refused: %u render targets (at most 8)", count);
        return refuse(bind_count_over_8);
    }
    const UINT listed = single_range ? (count ? 1 : 0) : count;
    if (listed && !x2w_local::readable(handles, listed * sizeof(*handles))) return refuse(bind_handles_unreadable);
    u64 stride = 0;
    if (single_range && count) {
        const UINT increment = the_device ? device_of(the_device)->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) : 0;
        stride = title_descriptor_stride(handles[0].ptr, increment);
        UINT available = 0;
        desktop_cpu_handle(handles[0].ptr, nullptr, &available);
        if (!stride || available < count) {
            if (first_calls(refused, 6))
                log("x2w-d3d12 OMSetRenderTargets refused: single range of %u at 0x%llx has %u descriptors left in its heap",
                    count, static_cast<unsigned long long>(handles[0].ptr), available);
            return refuse(bind_range_past_heap);
        }
    }
    D3D12_CPU_DESCRIPTOR_HANDLE desktop[8];
    ID3D12Resource* targets[8]{};
    DXGI_FORMAT formats[8]{};
    for (UINT i = 0; i < count; ++i) {
        const u64 title = single_range ? handles[0].ptr + i * stride : handles[i].ptr;
        desktop[i].ptr = static_cast<SIZE_T>(desktop_cpu_handle(title));
        const ResolvedView view = resolve_view(desktop[i].ptr, 20);
        if (view.wrong_kind && first_calls(refused, 6)) log("x2w-d3d12 OMSetRenderTargets refused: render target %u is not a render target view", i);
        targets[i] = view.resource;
        if (!targets[i]) {
            if (view.wrong_kind) return refuse(bind_wrong_kind);
            if (first_calls(refused, 6))
                log("x2w-d3d12 OMSetRenderTargets refused: render target %u of %u (title 0x%llx, %s) holds no known view", i, count,
                    static_cast<unsigned long long>(title), single_range ? "single range" : "array");
            InterlockedIncrement(&target_unresolved_at[i]);
            return refuse(bind_unresolved);
        }
        formats[i] = view_format_of(desktop[i].ptr, targets[i]);
    }
    D3D12_CPU_DESCRIPTOR_HANDLE desktop_dsv{};
    const D3D12_CPU_DESCRIPTOR_HANDLE* dsv_ptr = nullptr;
    ID3D12Resource* dsv_target = nullptr;
    DXGI_FORMAT dsv_format = DXGI_FORMAT_UNKNOWN;

    if (depth && !x2w_local::readable(depth, sizeof(*depth))) InterlockedIncrement(&target_depth_unreadable);
    if (depth && x2w_local::readable(depth, sizeof(*depth))) {
        const u64 resolved = desktop_cpu_handle(depth->ptr);
        if (!resolved && depth->ptr) InterlockedIncrement(&target_depth_unmapped);
        if (!resolved) {
            // The handle maps to no desktop descriptor: no depth is bound although the pipeline may expect one.
            InterlockedIncrement(&render_target_unresolved);
            if (first_calls(unresolved_logged, 6))
                log("x2w-d3d12 OMSetRenderTargets: depth handle 0x%llx maps to no descriptor; binding no depth", static_cast<unsigned long long>(depth->ptr));
        }
        if (resolved) {
            const ResolvedView view = resolve_view(resolved, 21);
            dsv_target = view.resource;
            if (!dsv_target) {
                InterlockedIncrement(&render_target_unresolved);
                if (first_calls(refused, 6))
                    log("x2w-d3d12 OMSetRenderTargets refused: depth handle is %s", view.wrong_kind ? "not a depth stencil view" : "not a known depth stencil view");
                return refuse(view.wrong_kind ? bind_depth_wrong_kind : bind_depth_unknown);
            }
            dsv_format = view_format_of(resolved, dsv_target);
            const bool read_depth = (view.dsv_flags & D3D12_DSV_FLAG_READ_ONLY_DEPTH) != 0;
            // A format without a stencil plane has no second plane to disagree with.
            const bool read_stencil = !format_has_stencil(dsv_format) ? read_depth : (view.dsv_flags & D3D12_DSV_FLAG_READ_ONLY_STENCIL) != 0;
            if (read_depth != read_stencil) InterlockedIncrement(&bound_mixed_depth);
            pass.read_depth_plane = read_depth;
            pass.read_stencil_plane = read_stencil;
            desktop_dsv.ptr = static_cast<SIZE_T>(resolved);
            dsv_ptr = &desktop_dsv;
        }
    }
    if (!count && !dsv_ptr) {
        if (depth && (!x2w_local::readable(depth, sizeof(*depth)) || depth->ptr)) return refuse(bind_depth_unknown);
        list->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
        pass.targets_refused = false;
        InterlockedIncrement(&target_binds[bind_empty]);
        return;
    }
    if (red_producers().enabled()) {
        std::copy_n(targets, 8, pass.trace_targets);
        pass.trace_depth = dsv_target;
    }
    for (UINT i = 0; i < count; ++i) {
        use_as_target(self, targets[i]);
        pass.target_formats[i] = formats[i];
        pass.targets[i] = targets[i];
        if (i == 0) pass.target0 = targets[i];
    }
    pass.dsv_resource = dsv_target;
    if (dsv_target) require_bound_depth(self);
    pass.target_count = count;
    pass.dsv_format = dsv_format;
    pass.dsv_read_only = pass.read_depth_plane && pass.read_stencil_plane;
    const auto single_sample = [](ID3D12Resource* resource) {
        const auto samples = resource->GetDesc().SampleDesc;
        return samples.Count == 1 && !samples.Quality;
    };
    pass.targets_single_sample = std::all_of(targets, targets + count, single_sample) &&
        (!dsv_target || single_sample(dsv_target));
    pass.depth_view_writable = dsv_target && !pass.read_depth_plane;
    list->OMSetRenderTargets(count, count ? desktop : nullptr, FALSE, dsv_ptr);
    pass.targets_bound = true;
    pass.targets_refused = false;
    InterlockedIncrement(&render_target_binds);
    InterlockedIncrement(&target_binds[single_range ? bind_single_range_bound : bind_bound]);
}
// One line per 300 presented frames: slot 50 outcomes since the last line (counters reset here).
void report_target_binds()
{
    LONG n[bind_outcomes], at[8];
    LONG calls = 0;
    for (int i = 0; i < bind_outcomes; ++i) calls += n[i] = InterlockedExchange(&target_binds[i], 0);
    for (int i = 0; i < 8; ++i) at[i] = InterlockedExchange(&target_unresolved_at[i], 0);
    log("x2w-targets 300 frames: binds %ld bound %ld single-range %ld outcomes (not recording %ld, count over 8 %ld, "
        "handles unreadable %ld, range past heap %ld, wrong kind %ld, unresolved target %ld [at %ld %ld %ld %ld %ld %ld %ld %ld], "
        "empty %ld, depth wrong kind %ld, depth unknown %ld, mixed depth/stencil bound %ld), bound without depth (unreadable %ld, "
        "unmapped %ld), draws without targets %ld; also dropped: clears (list closed or no colour %ld, no target %ld), "
        "viewports %ld, scissors %ld, blend factor %ld",
        calls, n[bind_bound] + n[bind_single_range_bound], n[bind_single_range_bound], n[bind_not_recording],
        n[bind_count_over_8], n[bind_handles_unreadable], n[bind_range_past_heap], n[bind_wrong_kind], n[bind_unresolved],
        at[0], at[1], at[2], at[3], at[4], at[5], at[6], at[7], n[bind_empty], n[bind_depth_wrong_kind],
        n[bind_depth_unknown], InterlockedExchange(&bound_mixed_depth, 0), InterlockedExchange(&target_depth_unreadable, 0),
        InterlockedExchange(&target_depth_unmapped, 0), InterlockedExchange(&draws_without_targets, 0),
        InterlockedExchange(&clears_dropped[0], 0), InterlockedExchange(&clears_dropped[1], 0), InterlockedExchange(&state_dropped[0], 0),
        InterlockedExchange(&state_dropped[1], 0), InterlockedExchange(&state_dropped[2], 0));
}
// Fixtures: slot 50 outcome counts since the last report, in TargetBind order (bound, single-range bound,
// then the refusal reasons). Returns how many outcomes exist.
extern "C" __declspec(dllexport) unsigned x2w_d3d12_target_bind_stats(unsigned* counts, unsigned capacity)
{
    for (unsigned i = 0; counts && i < capacity && i < bind_outcomes; ++i)
        counts[i] = static_cast<unsigned>(InterlockedCompareExchange(&target_binds[i], 0, 0));
    return bind_outcomes;
}
// List slots 21 / 22 / 23, RSSetViewports / RSSetScissorRects / OMSetBlendFactor.
void __stdcall list_viewports(Wrapper* self, UINT count, const D3D12_VIEWPORT* viewports)
{
    // Only a viewport that is forwarded counts as set; a call with no viewports leaves the state as it was.
    if (auto* list = recording(self); list && count && count <= 16 && x2w_local::readable(viewports, count * sizeof(*viewports))) {
        list->RSSetViewports(count, viewports);
        pass_of(self).viewport_set = true;
        pass_of(self).viewport0 = viewports[0];
        // LOCAL (video): a traced frame shows each viewport (letterbox questions).
        if (trace_frame_active)
            log("x2w-frame viewport list=%p count %u: x %.1f y %.1f w %.1f h %.1f", static_cast<void*>(self), count, viewports[0].TopLeftX,
                viewports[0].TopLeftY, viewports[0].Width, viewports[0].Height);
    } else if (list) InterlockedIncrement(&state_dropped[0]);
}
void __stdcall list_scissors(Wrapper* self, UINT count, const D3D12_RECT* rects)
{
    if (auto* list = recording(self); list && count && count <= 16 && x2w_local::readable(rects, count * sizeof(*rects))) {
        list->RSSetScissorRects(count, rects);
        pass_of(self).scissor_set = true;
        pass_of(self).scissor0 = rects[0];
        if (trace_frame_active)
            log("x2w-frame scissor list=%p count %u: %ld %ld %ld %ld", static_cast<void*>(self), count, rects[0].left, rects[0].top, rects[0].right,
                rects[0].bottom);
    } else if (list) InterlockedIncrement(&state_dropped[1]);
}
void __stdcall list_blend_factor(Wrapper* self, const float* factor)
{
    if (auto* list = recording(self); list && x2w_local::readable(factor, 16)) list->OMSetBlendFactor(factor);
    else if (list) InterlockedIncrement(&state_dropped[2]);
}
void __stdcall list_stencil_ref(Wrapper* self, UINT reference)
{
    if (auto* list = recording(self)) {
        pass_of(self).stencil_reference = reference;  // upstream X2W_DEPTH_STATE (#2169): list state the scoped reference restores
        list->OMSetStencilRef(reference & 0xFF);
    }
}
SRWLOCK cleared_depth_lock = SRWLOCK_INIT;
std::unordered_set<const ID3D12Resource*>& cleared_depth_targets()
{
    static std::unordered_set<const ID3D12Resource*> s;
    return s;
}
bool depth_target_cleared(const ID3D12Resource* resource)
{
    if (!resource) return false;
    AcquireSRWLockShared(&cleared_depth_lock);
    const bool found = cleared_depth_targets().count(resource) != 0;
    ReleaseSRWLockShared(&cleared_depth_lock);
    return found;
}
// LOCAL meshrt: the last forwarded clear of each depth resource (record time), for the traced-frame mesh lines.
struct DepthClearValue {
    float depth = 0;
    UINT stencil = 0, flags = 0;
    LONG frame = 0;
};
std::unordered_map<const ID3D12Resource*, DepthClearValue>& depth_clear_values()
{
    static std::unordered_map<const ID3D12Resource*, DepthClearValue> m;
    return m;
}
bool last_depth_clear(const ID3D12Resource* resource, DepthClearValue* out)
{
    if (!resource) return false;
    AcquireSRWLockShared(&cleared_depth_lock);
    const auto found = depth_clear_values().find(resource);
    const bool known = found != depth_clear_values().end();
    if (known) *out = found->second;
    ReleaseSRWLockShared(&cleared_depth_lock);
    return known;
}
LONG depth_clears[5];  // [0] forwarded, [1] list not open, [2] view not a known depth view, [3] depth value outside [0, 1], [4] nothing left to clear
void __stdcall list_clear_depth(Wrapper* self, u64 handle, UINT flags, float depth, UINT stencil, UINT rect_count, const D3D12_RECT* rects)
{
    InterlockedIncrement64(&list_slot_calls[51]);
    InterlockedIncrement64(&list_slot_census[51]);
    InterlockedIncrement(&draw_calls_seen);
    static LONG logged;
    const bool log_this = first_calls(logged, 8);
    auto* list = recording(self);
    if (!list) return (void)InterlockedIncrement(&depth_clears[1]);
    const u64 resolved = desktop_cpu_handle(handle);
    const ResolvedView view = resolved ? resolve_view(resolved, 21) : ResolvedView{};
    ID3D12Resource* target = view.resource;
    const DXGI_FORMAT format = target ? view_format_of(resolved, target) : DXGI_FORMAT_UNKNOWN;
    UINT clear = flags & (D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL);
    if (!format_has_stencil(format)) clear &= ~D3D12_CLEAR_FLAG_STENCIL;
    if (log_this)
        log("x2w-render ClearDepthStencilView view=0x%llx target=%p format %d flags 0x%x depth %g stencil %u rects %u", handle,
            static_cast<void*>(target), static_cast<int>(format), flags, depth, stencil & 0xFF, rect_count);
    if (!target) return (void)InterlockedIncrement(&depth_clears[2]);
    if ((clear & D3D12_CLEAR_FLAG_DEPTH) && !(depth >= 0.0f && depth <= 1.0f)) return (void)InterlockedIncrement(&depth_clears[3]);
    if (!clear) return (void)InterlockedIncrement(&depth_clears[4]);
    if (rect_count && !x2w_local::readable(rects, rect_count * sizeof(*rects))) return (void)InterlockedIncrement(&depth_clears[2]);
    const auto* tracked = pass_of(self).states.find(target);
    const bool known = tracked != nullptr;
    const auto before = known ? tracked->depth : D3D12_RESOURCE_STATE_COMMON;
    const auto before_stencil = known ? tracked->stencil : before;
    require_state(self, target, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    list->ClearDepthStencilView(D3D12_CPU_DESCRIPTOR_HANDLE{static_cast<SIZE_T>(resolved)}, static_cast<D3D12_CLEAR_FLAGS>(clear), depth,
                                static_cast<UINT8>(stencil & 0xFF), rect_count, rect_count ? rects : nullptr);
    if (known) require_planes(self, target, before, before_stencil);
    // Only a depth clear initialises the depth plane; marked at record time, never unmarked (a freed and reused address stays marked).
    if (clear & D3D12_CLEAR_FLAG_DEPTH) {
        AcquireSRWLockExclusive(&cleared_depth_lock);
        cleared_depth_targets().insert(target);
        ReleaseSRWLockExclusive(&cleared_depth_lock);
    }
    {  // LOCAL meshrt: the value, for the traced-frame mesh lines (one map store per forwarded clear)
        AcquireSRWLockExclusive(&cleared_depth_lock);
        DepthClearValue& v = depth_clear_values()[target];
        if (clear & D3D12_CLEAR_FLAG_DEPTH) v.depth = depth;
        if (clear & D3D12_CLEAR_FLAG_STENCIL) v.stencil = stencil & 0xFF;
        v.flags = clear;
        v.frame = frame_number;
        ReleaseSRWLockExclusive(&cleared_depth_lock);
    }
    InterlockedIncrement(&depth_clears[0]);
}
// List slot 24 (upstream's OMSetStencilRef handler is merged into the local one above: it records the reference
// for X2W_DEPTH_STATE's scoped stencil reference in d3d12_draw.inl).
// List slot 43 (desktop 39 + 4), set a compute root SRV backed by this recording's upload pages.
void __stdcall list_compute_upload_srv(Wrapper* self, UINT parameter, UINT64 address)
{
    auto* list = recording(self);
    auto& pass = pass_of(self);
    const auto* params = pass.compute_params;
    if (!list || !params || parameter >= params->size() || parameter >= pass.desktop_compute_upload_srvs.size() ||
        (*params)[parameter].type != D3D12_ROOT_PARAMETER_TYPE_SRV) return;
    const auto gpu = list_upload_gpu(self, address, 1);
    if (gpu && !(gpu % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT)) {
        pass.desktop_compute_upload_srvs[parameter] = gpu;
        list->SetComputeRootShaderResourceView(parameter, gpu);
    }
}

// List slot 44 (desktop 40 + 4), SetGraphicsRootShaderResourceView(parameter, title GPU address).
bool root_srv_lazy()
{
    static const bool on = x2w_local::env_on("X2W_ROOT_SRV_LAZY");
    return on;
}
LONG root_srv_deferred, root_srv_lazy_bound;
void bind_root_srv(Wrapper* self, ID3D12GraphicsCommandList* list, UINT parameter, u64 address);
void __stdcall list_root_srv(Wrapper* self, UINT parameter, u64 address)
{
    auto* list = recording(self);
    Pass& pass = pass_of(self);
    if (parameter < 32) pass.srv_bound &= ~(1u << parameter), pass.srv_pending &= ~(1u << parameter);
    if (parameter < 16) pass.root_address[parameter] = address;
    // Under a deferred heap-indexed root the desktop list holds another root signature; only the title address is kept.
    if (!list || !pass.root || pass.root_heap_flags || parameter >= pass.root->size() || parameter >= 32 ||
        (*pass.root)[parameter].type != D3D12_ROOT_PARAMETER_TYPE_SRV)
        return;
    if (pass.indirect_graphics.cached_mesh && pass.indirect_graphics.cached_srv_root == parameter)
        return; // The slot 61 draw binds the validated lease rather than an upload snapshot.
    if (const auto gpu = list_upload_gpu(self, address, 1)) {
        if (gpu % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT) return;
        list->SetGraphicsRootShaderResourceView(parameter, gpu);
        pass.srv_bound |= 1u << parameter;
        return;
    }
    if (root_srv_lazy() && parameter < 16) {
        pass.srv_pending |= 1u << parameter;
        const LONG n = InterlockedIncrement(&root_srv_deferred);
        if (n == 1) log("x2w-render X2W_ROOT_SRV_LAZY=1: root SRVs are copied at a real-pipeline draw only");
        return;
    }
    bind_root_srv(self, list, parameter, address);
}
// X2W_ROOT_SRV_LAZY: before a draw on a real desktop pipeline, the root SRVs set since the root signature
// was bound and not copied yet.
void bind_lazy_root_srvs(Wrapper* self, ID3D12GraphicsCommandList* list)
{
    if (!root_srv_lazy()) return;
    Pass& pass = pass_of(self);
    if (!pass.root) return;
    for (UINT p = 0; p < 16 && p < pass.root->size(); ++p)
        if ((pass.srv_pending & (1u << p)) && (*pass.root)[p].type == D3D12_ROOT_PARAMETER_TYPE_SRV && pass.root_address[p]) {
            pass.srv_pending &= ~(1u << p);
            bind_root_srv(self, list, p, pass.root_address[p]);
            const LONG n = InterlockedIncrement(&root_srv_lazy_bound);
            if (x2w_local::first_of(n, 3))
                log("x2w-render X2W_ROOT_SRV_LAZY: root SRV %u copied at a real-pipeline draw (%ld so far; %ld calls deferred)", p, n,
                    root_srv_deferred);
        }
}
void bind_root_srv(Wrapper* self, ID3D12GraphicsCommandList* list, UINT parameter, u64 address)
{
    Pass& pass = pass_of(self);
    const std::size_t bytes = readable_window(address, 0x10000);
    const D3D12_GPU_VIRTUAL_ADDRESS desktop = bytes ? upload_from_title(address, bytes) : 0;
    static LONG logged;
    if (first_calls(logged, 4)) {
        char head[3 * 32 + 1];
        hex_text(reinterpret_cast<const void*>(address), 32, head, sizeof(head));
        log("x2w-render root SRV %u at title 0x%llx: %zu bytes -> desktop 0x%llx [%s]", parameter, address, bytes, desktop, head);
    }
    if (!desktop) return;
    list->SetGraphicsRootShaderResourceView(parameter, desktop);
    pass.srv_bound |= 1u << parameter;
}
// LOCAL: list slot 61, ExecuteIndirect(signature, max count, argument resource, offset, count resource,
// count offset). Not executed yet. During a traced frame every call is recorded (signature, the bound
// pipeline's creation path, argument address) with whether its record was non-zero when the list was
// recorded; at the end of the frame the records are read again and summarised per group. Words that
// only the console GPU would write stay as they were, so "non-zero at end" means the CPU wrote them.
LONG indirect_calls_local;  // LOCAL trace (upstream has indirect_calls[][] by outcome)
struct IndirectCall {
    const void* signature;
    int path;
    const void* base;
    const unsigned char* record;
    UINT stride;
    UINT type;
    bool nonzero_at_record;
};
SRWLOCK indirect_lock = SRWLOCK_INIT;
std::vector<IndirectCall>& indirect_trace()
{
    static std::vector<IndirectCall> calls;
    return calls;
}
void indirect_execute(Wrapper* self, void* signature, UINT max_count, Wrapper* arguments, u64 offset, Wrapper* count_resource, u64 count_offset);
// LOCAL (scene): X2W_MESH_RECORD_TRACE (d3d12_record_trace.inl), traced frame only.
void record_trace_indirect(Wrapper* self, UINT type, UINT stride, UINT max_count, Wrapper* arguments, u64 offset);
void record_trace_dispatch(Wrapper* self, u64 key, UINT x, UINT y, UINT z, bool executed, int outcome);
void record_trace_report(ID3D12CommandQueue* queue);
void record_trace_mesh_read(u64 key, UINT srv, int source, int a, int b, u64 address, u64 bytes);  // d3d12_draw.inl, mesh calls
void record_trace_mesh_heap(Wrapper* self, const Translation& tr);
void __stdcall list_execute_indirect(Wrapper* self, void* signature, UINT max_count, Wrapper* arguments, u64 offset,
                                     Wrapper* count_resource, u64 count_offset)
{
    InterlockedIncrement(&indirect_calls_local);
    indirect_execute(self, signature, max_count, arguments, offset, count_resource, count_offset);  // LOCAL: d3d12_indirect.inl
    if (!trace_frame_active) return;
    SignatureInfo info;
    AcquireSRWLockShared(&signature_lock);
    const auto found = signature_infos().find(signature);
    if (found != signature_infos().end()) info = found->second;
    ReleaseSRWLockShared(&signature_lock);
    IndirectCall call{};
    call.signature = signature;
    call.stride = info.stride;
    call.type = info.args.empty() ? ~0u : static_cast<UINT>(info.args.back().Type);
    const PipelineInfo* pipeline = pass_of(self).pipeline ? pipeline_info(pass_of(self).pipeline) : nullptr;
    call.path = pipeline ? pipeline->path : -1;
    call.base = arguments && arguments->own().kind == resource_kind ? arguments->own().title_memory : nullptr;
    call.record = call.base ? static_cast<const unsigned char*>(call.base) + offset : nullptr;
    const UINT bytes = info.stride && info.stride <= 64 ? info.stride : 12;
    if (call.record && x2w_local::readable(call.record, bytes))
        for (UINT i = 0; i < bytes; ++i) call.nonzero_at_record |= call.record[i] != 0;
    else
        call.record = nullptr;
    (void)max_count;
    (void)count_resource;
    (void)count_offset;
    AcquireSRWLockExclusive(&indirect_lock);
    indirect_trace().push_back(call);
    ReleaseSRWLockExclusive(&indirect_lock);
    record_trace_indirect(self, call.type, info.stride, max_count, arguments, offset);  // LOCAL (scene)
}
void summarise_indirect_calls()
{
    record_trace_report(nullptr);  // LOCAL (scene): no-op when the busy-frame end already reported
    AcquireSRWLockExclusive(&indirect_lock);
    std::vector<IndirectCall> calls;
    calls.swap(indirect_trace());
    ReleaseSRWLockExclusive(&indirect_lock);
    struct Group {
        int calls = 0, unreadable = 0, nonzero_at_record = 0, nonzero_at_end = 0;
        UINT stride = 0, type = 0;
        std::uint32_t sample[4] = {};
    };
    std::map<std::tuple<const void*, int, const void*>, Group> groups;
    for (const auto& c : calls) {
        Group& g = groups[{c.signature, c.path, c.base}];
        ++g.calls;
        g.stride = c.stride;
        g.type = c.type;
        if (!c.record) {
            ++g.unreadable;
            continue;
        }
        const UINT bytes = c.stride && c.stride <= 64 ? c.stride : 12;
        bool nonzero = false;
        if (x2w_local::readable(c.record, bytes))
            for (UINT i = 0; i < bytes; ++i) nonzero |= c.record[i] != 0;
        g.nonzero_at_record += c.nonzero_at_record;
        if (nonzero) {
            if (!g.nonzero_at_end) std::memcpy(g.sample, c.record, bytes < 16 ? bytes : 16);
            ++g.nonzero_at_end;
        }
    }
    log("x2w-frame indirect summary: %zu calls, %zu groups (signature, pipeline creation path, argument resource)", calls.size(), groups.size());
    for (const auto& [key, g] : groups)
        log("x2w-frame indirect group sig %p type %u stride %u path %d args %p: calls %d, unreadable %d, non-zero at record %d, at end %d, sample %08x %08x %08x %08x",
            std::get<0>(key), g.type, g.stride, std::get<1>(key), std::get<2>(key), g.calls, g.unreadable, g.nonzero_at_record,
            g.nonzero_at_end, g.sample[0], g.sample[1], g.sample[2], g.sample[3]);
}
// List slot 42 (desktop 38 + 4), SetGraphicsRootConstantBufferView(parameter, title GPU address).
// A translated shader reads a root CBV through the same pointer source as a root SRV (root_address). Nothing
// is forwarded to the desktop list and srv_bound is left alone: bit 0 means a vertex-fetch SRV is bound.
void __stdcall list_root_cbv(Wrapper* self, UINT parameter, u64 address)
{
    auto* list = recording(self);
    if (!list) return;
    Pass& pass = pass_of(self);
    if (!pass.root || parameter >= pass.root->size() || parameter >= 16) return;
    const D3D12_ROOT_PARAMETER_TYPE type = (*pass.root)[parameter].type;
    if (type != D3D12_ROOT_PARAMETER_TYPE_CBV) {
        static LONG refused;
        if (first_calls(refused, 1))
            log("x2w-render root CBV %u ignored: the root signature declares parameter type %d", parameter, static_cast<int>(type));
        return;
    }
    pass.root_address[parameter] = (address && readable_window(address, 1)) ? address : 0;
}
extern LONG translated_draws, untranslated_draws, frame_number;
// LOCAL: d3d12_indirect.inl, the indirect call at the issue points of the dispatch and draw paths.
bool indirect_pending(Wrapper* self);
bool indirect_draw_issue(Wrapper* self, ID3D12GraphicsCommandList* list, bool indexed);
void indirect_dispatch(Wrapper* self, ID3D12GraphicsCommandList* list, UINT x, UINT y, UINT z);
void mesh_note57(const void* desc, const Wrapper* self);  // LOCAL meshrt: d3d12_mesh.inl
bool draw_translated(Wrapper* self, bool indexed, UINT count, UINT instances, UINT start, INT base_vertex, UINT start_instance);
extern thread_local u64 last_draw_key;          // LOCAL (scene): defined in d3d12_draw.inl, for the traced-frame draw lines
extern thread_local const char* last_draw_why;
extern thread_local u64 last_draw_key;
extern thread_local const char* last_draw_why;
// List slot 12, DrawInstanced.
void __stdcall list_draw(Wrapper* self, UINT vertices, UINT instances, UINT first_vertex, UINT first_instance)
{
    auto* list = recording(self);
    Pass& pass = pass_of(self);
    if (list && !pass.pipeline_real) {
        const bool done = draw_translated(self, false, vertices, instances, first_vertex, 0, first_instance);
        InterlockedIncrement(done ? &translated_draws : &untranslated_draws);
        // LOCAL (scene): restored from f5e0f425 (lost in a merge), as in list_draw_indexed.
        if (trace_frame_active) {
            note_traced_target(self, pass.target0, done);
            const D3D12_RESOURCE_DESC rd = pass.target0 ? pass.target0->GetDesc() : D3D12_RESOURCE_DESC{};
            log("x2w-frame draw %u %s target %p %llux%u fmt %d topology 0x%x key %016llx %s", vertices, done ? "drawn" : "NOT drawn",
                static_cast<void*>(pass.target0), static_cast<unsigned long long>(rd.Width), rd.Height, rd.Format, pass.topology,
                last_draw_key, done ? "" : (last_draw_why ? last_draw_why : "draw failed"));
        }
        return;
    }
    if (list && pass.root_heap_flags) {
        // The desktop list holds no copy of the title's heaps, and the root constants and descriptor tables the
        // title set are kept only as title values, so the deferred root signature cannot be set with its state.
        x2w_startup::RealRootDraw request;
        request.root_flags = pass.root_heap_flags;
        request.state_restorable = false;
        const x2w_startup::RealRootDecision decision = x2w_startup::decide_real_root(request);
        if (decision != x2w_startup::RealRootDecision::bind) {
            const LONG n = InterlockedIncrement(&heap_root_refusals[static_cast<int>(decision)]);
            InterlockedIncrement(&render_draws_skipped);
            if (n <= 3 || n % 1000 == 0)
                log("x2w-render real draw refused: root signature heap flags 0x%x, %s (%ld so far)", pass.root_heap_flags,
                    x2w_startup::real_root_reason(decision), n);
            return;
        }
        // Not reached with the facts above: a change that mirrors the title heaps and restores the root state on the
        // list, with an end-to-end fixture, makes the deferred bind here.
        list->SetGraphicsRootSignature(pass.deferred_root);
    }
    // The translated vertex shaders read their fetch table through root parameter 0.
    if (list && pass.pipeline_real) bind_lazy_root_srvs(self, list);  // LOCAL stall: X2W_ROOT_SRV_LAZY
    const bool real_drawn = list && pass.pipeline_real && pass.targets_bound && pass.topology_ok && pass.topology != 0x11 && (pass.srv_bound & 1);
    if (real_drawn) {
        const auto red_samples = red_producer_draw_before(self, list, "native-draw", 0);
        list->DrawInstanced(vertices, instances, first_vertex, first_instance);
        red_producer_draw_after(self, red_samples);
        const LONG n = InterlockedIncrement(&render_draws);
        if (x2w_local::first_of(n, 3)) log("x2w-render DrawInstanced(%u, %u, %u, %u) recorded", vertices, instances, first_vertex, first_instance);
    } else {
        InterlockedIncrement(&render_draws_skipped);
    }
    // LOCAL (scene): the desktop-pipeline path in a traced frame, so no draw of the frame goes unlogged.
    if (trace_frame_active) {
        note_traced_target(self, pass.target0, real_drawn);
        const D3D12_RESOURCE_DESC rd = pass.target0 ? pass.target0->GetDesc() : D3D12_RESOURCE_DESC{};
        log("x2w-frame draw %u %s target %p %llux%u fmt %d topology 0x%x desktop pipeline%s", vertices, real_drawn ? "drawn" : "NOT drawn",
            static_cast<void*>(pass.target0), static_cast<unsigned long long>(rd.Width), rd.Height, rd.Format, pass.topology,
            real_drawn ? "" : !list ? ": not recording" : !pass.pipeline_real ? ": no pipeline" : ": state (targets/topology/srv 0)");
    }
}

extern SRWLOCK written_lock;
struct TextureWrite {
    u64 generation = 0;
    UINT bc7_width = 0, bc7_height = 0;
    bool desktop_rows = false;  // the last XG copy wrote desktop-footprint rows (not the opt-in BC7 tiles)
};
std::unordered_map<u64, TextureWrite>& written_textures();
ID3D12Resource* sampled_address_texture(Wrapper* resource);
bool console_layout_srv(const D3D12_RESOURCE_DESC& desc);
#include "startup_stubs/d3d12_draw.inl"
#include "startup_stubs/d3d12_regblob.inl"


LONG frame_dispatches;  // dispatches since the last present (LOCAL instrumentation)
// LOCAL: slot 67 register blobs ([1]/[2] and [3]/[4] of the description), copied at creation because the
// title frees them; written next to the code as req_c67_<key>.regs (length-prefixed pair).
std::unordered_map<Wrapper*, std::vector<unsigned char>>& compute_regs() { static std::unordered_map<Wrapper*, std::vector<unsigned char>> m; return m; }
// LOCAL: compute pipeline wrapper -> req_cs key, and dispatch counts per key (instrumentation).
SRWLOCK compute_lock = SRWLOCK_INIT;
std::unordered_map<Wrapper*, u64>& compute_keys() { static std::unordered_map<Wrapper*, u64> m; return m; }
// LOCAL: compute root state the title sets (list slots = desktop + 4), replayed on a real dispatch.
D3D12_GPU_VIRTUAL_ADDRESS desktop_va(u64 address);
void note_window(ID3D12Resource* buffer);
bool execute_translated(Wrapper* self, ID3D12GraphicsCommandList* list, u64 key, UINT x, UINT y, UINT z);
bool rawaddr_pipeline(u64 key, Wrapper* root, Wrapper* pipeline);                              // d3d12_rawaddr.inl
bool rawaddr_dispatch(Wrapper* self, ID3D12GraphicsCommandList* list, UINT x, UINT y, UINT z);  // d3d12_rawaddr.inl
SRWLOCK compute_pso_lock = SRWLOCK_INIT;
std::unordered_map<Wrapper*, ID3D12PipelineState*>& compute_psos()
{
    static std::unordered_map<Wrapper*, ID3D12PipelineState*> m;
    return m;
}
LONG compute_executed, compute_skipped_args;
// LOCAL: dispatch outcomes since the last 300-frame report, by path and reason (report_dispatches).
enum ComputeOutcome { co_c62_run, co_c62_table_arg, co_c62_unmapped, co_c67_run, co_c67_off, co_c67_other_key, co_c67_untranslated,
                      co_c67_heap, co_not_run, co_cx_skip, co_rawaddr_run, co_c67_unresolved, co_c67_compiling /* X2W_CX_ASYNC */, co_count };
LONG compute_outcomes[co_count];    // compute_lock; counted with the per-key count by execute_dispatch
thread_local int dispatch_outcome;  // set on this thread by the path that handled the dispatch

// LOCAL: the dispatches issued on each desktop list (Dispatch or ExecuteIndirect, through indirect_dispatch), in
// order, since its last Reset, plus the recording before that one (a list can be reset while its previous
// submission still runs). The DRED report maps the in-progress Dispatch op to one of them.
struct IssuedDispatch { u64 key; UINT x, y, z; char path; bool indirect; };
struct IssuedList { std::vector<IssuedDispatch> current, previous; };
thread_local u64 cx_issue_key;    // set by execute_dispatch for indirect_dispatch
thread_local char cx_issue_path;  // 'c' slot-62 desktop pipeline, 't' translated, 'r' raw-address
SRWLOCK issued_lock = SRWLOCK_INIT;
constexpr std::size_t issued_cap = 16384;  // per recording
std::unordered_map<const void*, IssuedList>& issued_lists() { static std::unordered_map<const void*, IssuedList> m; return m; }
const void* issued_id(ID3D12GraphicsCommandList* list) { return static_cast<const void*>(static_cast<ID3D12CommandList*>(list)); }
void note_issued_dispatch(ID3D12GraphicsCommandList* list, UINT x, UINT y, UINT z, bool indirect)
{
    AcquireSRWLockExclusive(&issued_lock);
    std::vector<IssuedDispatch>& v = issued_lists()[issued_id(list)].current;
    if (v.size() < issued_cap) v.push_back({cx_issue_key, x, y, z, cx_issue_path, indirect});
    ReleaseSRWLockExclusive(&issued_lock);
}
void forget_list_dispatches(ID3D12GraphicsCommandList* list)
{
    AcquireSRWLockExclusive(&issued_lock);
    const auto found = issued_lists().find(issued_id(list));
    if (found != issued_lists().end()) {
        found->second.previous.swap(found->second.current);
        found->second.current.clear();
    }
    ReleaseSRWLockExclusive(&issued_lock);
}
const char* issued_path_name(char path)
{
    return path == 'c' ? "slot-62" : path == 't' ? "translated" : path == 'r' ? "raw-address" : path == 'g' ? "indirect guard pass" : path == 'b' ? "GPU buffer copy (draw-rt)" : "?";
}
// For one incomplete DRED list: the dispatch at op `done` (the Dispatch ops before it are counted from the op
// types, so the barriers between dispatches need no fixed stride) and the five before and after it.
void report_dred_dispatches(const D3D12_AUTO_BREADCRUMB_NODE* node, UINT done)
{
    if (!node->pCommandList || !node->pCommandHistory) return;
    UINT before = 0, total = 0;  // Dispatch ops before op `done`, and in the whole list
    for (UINT i = 0; i < node->BreadcrumbCount; ++i)
        if (node->pCommandHistory[i] == D3D12_AUTO_BREADCRUMB_OP_DISPATCH) {
            if (i < done) ++before;
            ++total;
        }
    const bool at_dispatch = done < node->BreadcrumbCount && node->pCommandHistory[done] == D3D12_AUTO_BREADCRUMB_OP_DISPATCH;
    std::vector<IssuedDispatch> current, previous;
    AcquireSRWLockShared(&issued_lock);
    const auto found = issued_lists().find(issued_id(node->pCommandList));
    if (found != issued_lists().end()) current = found->second.current, previous = found->second.previous;
    ReleaseSRWLockShared(&issued_lock);
    const auto direct_of = [](const std::vector<IssuedDispatch>& v) {
        std::vector<IssuedDispatch> out;
        for (const IssuedDispatch& d : v)
            if (!d.indirect) out.push_back(d);
        return out;
    };
    std::vector<IssuedDispatch> direct = direct_of(current);
    const char* which = "current recording";
    if (direct.size() != total) {
        std::vector<IssuedDispatch> older = direct_of(previous);
        if (older.size() == total) direct.swap(older), which = "previous recording";
    }
    const bool match = direct.size() == total;
    log("x2w-dred dispatches list %p: op %u is %s; %u Dispatch ops before it, %u in the list; %zu direct dispatches in the %s "
        "(%zu recorded now, %zu before the last Reset)%s",
        static_cast<const void*>(node->pCommandList), done, at_dispatch ? "a Dispatch" : "not a Dispatch", before, total, direct.size(),
        which, current.size(), previous.size(), match ? "" : "; COUNT MISMATCH, the mapping below is not reliable");
    if (direct.empty()) return;
    const std::size_t at = before < direct.size() ? before : direct.size() - 1;
    const std::size_t from = at > 5 ? at - 5 : 0, to = at + 6 < direct.size() ? at + 6 : direct.size();
    for (std::size_t i = from; i < to; ++i)
        log("x2w-dred   %s dispatch %zu: key %016llx (%u, %u, %u) %s", i == at && at_dispatch ? ">>" : "  ", i, direct[i].key,
            direct[i].x, direct[i].y, direct[i].z, issued_path_name(direct[i].path));
}

// LOCAL: X2W_CX_SKIP=<key>[,<key>...] (hex): dispatches of these keys are skipped (outcome "X2W_CX_SKIP").
bool cx_skip_key(u64 key)
{
    static const std::vector<u64> keys = [] {
        std::vector<u64> out;
        char value[2048]{};
        const DWORD n = GetEnvironmentVariableA("X2W_CX_SKIP", value, sizeof(value));
        if (!n || n >= sizeof(value)) return out;
        for (const char* p = value; *p;) {
            char* end = nullptr;
            const u64 k = std::strtoull(p, &end, 16);
            if (end == p) {
                ++p;
                continue;
            }
            out.push_back(k);
            p = end;
        }
        log("x2w-cx X2W_CX_SKIP: %zu keys", out.size());
        for (const u64 k : out) log("x2w-cx X2W_CX_SKIP key %016llx", k);
        return out;
    }();
    if (keys.empty()) return false;
    return std::find(keys.begin(), keys.end(), key) != keys.end();
}
// slot 33, SetComputeRootSignature(root)
// LOCAL: as in D3D12, setting the root signature already bound keeps the root arguments; another one clears them
// (X2W_CX_ROOT_RESET=1: every call clears them, as before).
LONG compute_root_same, compute_root_changed;
void __stdcall list_compute_root(Wrapper* self, Wrapper* root)
{
    static const bool always_reset = x2w_local::env_on("X2W_CX_ROOT_RESET");  // the old behaviour
    Pass& pass = pass_of(self);
    if (root && root == pass.compute_root && !always_reset) {
        InterlockedIncrement(&compute_root_same);
        return;
    }
    InterlockedIncrement(&compute_root_changed);
    pass.compute_root = root;
    for (auto& k : pass.compute_kind) k = 0;
    for (auto& s : pass.compute_set) s = 0;
}
// slot 35, SetComputeRootDescriptorTable(parameter, handle): not replayed (the executed shaders use none).
void __stdcall list_compute_table(Wrapper* self, UINT parameter, u64 gpu_handle)
{
    if (parameter >= 16) return;
    u64 address = 0;  // the table's address in the title's descriptor memory (as list_table)
    AcquireSRWLockShared(&note_lock);
    for (const HeapInfo& heap : heap_infos())
        if (heap.desktop_gpu && heap.title_cpu && gpu_handle >= heap.desktop_gpu && gpu_handle < heap.desktop_gpu + u64{heap.count} * heap.increment)
            address = heap.title_cpu + (gpu_handle - heap.desktop_gpu);
    ReleaseSRWLockShared(&note_lock);
    pass_of(self).compute_kind[parameter] = 5;
    pass_of(self).compute_address[parameter] = address;
    pass_of(self).compute_set[parameter] = 0;
}
// slot 37, SetComputeRoot32BitConstant(parameter, value, offset)
void __stdcall list_compute_constant(Wrapper* self, UINT parameter, UINT value, UINT offset)
{
    Pass& pass = pass_of(self);
    if (parameter >= 16 || offset >= 16) return;
    if (pass.compute_kind[parameter] != 1) pass.compute_set[parameter] = 0;
    pass.compute_kind[parameter] = 1;
    pass.compute_constants[parameter][offset] = value;
    pass.compute_set[parameter] |= static_cast<std::uint16_t>(1u << offset);
    if (offset + 1 > pass.compute_count[parameter]) pass.compute_count[parameter] = offset + 1;
}
// slot 39, SetComputeRoot32BitConstants(parameter, count, data, offset)
void __stdcall list_compute_constants(Wrapper* self, UINT parameter, UINT count, const void* data, UINT offset)
{
    Pass& pass = pass_of(self);
    if (parameter >= 16 || offset >= 16 || !data || !x2w_local::readable(data, std::size_t{count} * 4)) return;
    if (offset + count > 16) count = 16 - offset;
    if (pass.compute_kind[parameter] != 1) pass.compute_set[parameter] = 0;
    pass.compute_kind[parameter] = 1;
    std::memcpy(&pass.compute_constants[parameter][offset], data, std::size_t{count} * 4);
    for (UINT i = 0; i < count; ++i) pass.compute_set[parameter] |= static_cast<std::uint16_t>(1u << (offset + i));
    if (offset + count > pass.compute_count[parameter]) pass.compute_count[parameter] = offset + count;
}
void compute_root_descriptor(Wrapper* self, UINT parameter, u64 address, int kind)
{
    Pass& pass = pass_of(self);
    if (parameter >= 16) return;
    pass.compute_kind[parameter] = kind;
    pass.compute_address[parameter] = address;
    pass.compute_set[parameter] = 0;
}
void __stdcall list_compute_cbv(Wrapper* self, UINT parameter, u64 address) { compute_root_descriptor(self, parameter, address, 2); }
void __stdcall list_compute_srv(Wrapper* self, UINT parameter, u64 address) { compute_root_descriptor(self, parameter, address, 3); }
void __stdcall list_compute_uav(Wrapper* self, UINT parameter, u64 address) { compute_root_descriptor(self, parameter, address, 4); }

// Runs a dispatch through a desktop compute pipeline when one exists for the bound pipeline.
bool execute_dispatch_inner(Wrapper* self, UINT x, UINT y, UINT z, u64 key)
{
    static const bool enabled = !x2w_local::env_on("X2W_COMPUTE_OFF");
    if (cx_skip_key(key)) {
        dispatch_outcome = co_cx_skip;
        static LONG logged;
        if (first_calls(logged, 8)) log("x2w-cx dispatch of key %016llx skipped (X2W_CX_SKIP)", key);
        return false;
    }
    auto* list = recording(self);
    Pass& pass = pass_of(self);
    if (!enabled || !list || !pass.pipeline) {
        dispatch_outcome = co_not_run;
        return false;
    }
    cx_issue_key = key;
    cx_issue_path = 'r';
    if (rawaddr_dispatch(self, list, x, y, z)) {  // LOCAL raw-address shaders (d3d12_rawaddr.inl)
        dispatch_outcome = co_rawaddr_run;
        return true;
    }
    cx_issue_path = 't';
    if (!pass.compute_root || !pass.compute_root->own().real) return execute_translated(self, list, key, x, y, z);
    AcquireSRWLockShared(&compute_pso_lock);
    const auto found = compute_psos().find(pass.pipeline);
    ID3D12PipelineState* pso = found != compute_psos().end() ? found->second : nullptr;
    ReleaseSRWLockShared(&compute_pso_lock);
    if (!pso) return execute_translated(self, list, key, x, y, z);
    cx_issue_path = 'c';
    D3D12_GPU_VIRTUAL_ADDRESS addresses[16] = {};
    for (UINT i = 0; i < 16; ++i) {
        const int kind = pass.compute_kind[i];
        if (kind == 5) {
            InterlockedIncrement(&compute_skipped_args);
            dispatch_outcome = co_c62_table_arg;
            return false;
        }
        if (kind >= 2 && kind <= 4) {
            addresses[i] = desktop_va(pass.compute_address[i]);
            if (!addresses[i] && pass.compute_address[i]) {
                InterlockedIncrement(&compute_skipped_args);
                dispatch_outcome = co_c62_unmapped;
                return false;
            }
        }
    }
    list->SetPipelineState(pso);
    list->SetComputeRootSignature(static_cast<ID3D12RootSignature*>(pass.compute_root->own().real));
    for (UINT i = 0; i < 16; ++i) {
        switch (pass.compute_kind[i]) {
        case 1: list->SetComputeRoot32BitConstants(i, pass.compute_count[i], pass.compute_constants[i], 0); break;
        case 2: list->SetComputeRootConstantBufferView(i, addresses[i]); break;
        case 3: list->SetComputeRootShaderResourceView(i, addresses[i]); break;
        case 4: list->SetComputeRootUnorderedAccessView(i, addresses[i]); break;
        default: break;
        }
    }
    gpu_time::mark(self, list, gpu_time::c_c62, key, false);  // LOCAL: X2W_GPU_TIME_CLASSES
    indirect_dispatch(self, list, x, y, z);  // LOCAL: Dispatch, or the pending ExecuteIndirect (d3d12_indirect.inl)
    gpu_time::mark(self, list, gpu_time::c_c62, key, true);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    list->ResourceBarrier(1, &barrier);
    const LONG n = InterlockedIncrement(&compute_executed);
    dispatch_outcome = co_c62_run;
    if (x2w_local::first_of(n, 4) || n % 5000 == 0) log("x2w-compute dispatch %ld executed (%u, %u, %u)", n, x, y, z);
    return true;
}
struct DispatchCount { LONG64 calls = 0; LONG64 groups = 0; UINT x = 0, y = 0, z = 0; };
std::unordered_map<u64, DispatchCount>& dispatch_counts() { static std::unordered_map<u64, DispatchCount> m; return m; }
// LOCAL: every dispatch (list slot 14 and an ExecuteIndirect dispatch) is counted per key and by outcome in one
// compute_lock hold, so the per-key list and the outcome line of one 300-frame report cover the same dispatches.
// An indirect dispatch passes (0, 0, 0): it adds a call but keeps the key's last direct group counts.
bool execute_dispatch(Wrapper* self, UINT x, UINT y, UINT z, u64 key)
{
    dispatch_outcome = co_not_run;
    const bool executed = execute_dispatch_inner(self, x, y, z, key);
    const int outcome = dispatch_outcome >= 0 && dispatch_outcome < co_count ? dispatch_outcome : co_not_run;
    if (trace_frame_active) record_trace_dispatch(self, key, x, y, z, executed, outcome);  // LOCAL (scene)
    AcquireSRWLockExclusive(&compute_lock);
    DispatchCount& count = dispatch_counts()[key];
    ++count.calls;
    count.groups += LONG64{x} * y * z;
    if (x | y | z) count.x = x, count.y = y, count.z = z;
    ++compute_outcomes[outcome];
    ReleaseSRWLockExclusive(&compute_lock);
    return executed;
}
// LOCAL: slot-67 request files req_c67_<key>.{cs,meta,root,desc,regs}, then the marker .done. Once per key per run
// (when the key is first computed for a pipeline, not per dispatch): a key with .done costs one GetFileAttributesW,
// as the old .cs check did. Otherwise each part is checked; a part that is absent or has another size than its data
// (a write cut off by an exit) is written through <name>.<pid>.<tid>.tmp and a rename (regblob::write_whole), and .done
// is written last only when all five parts are on disk. A set started by an older build (no .regs) or cut off
// mid-write (only .cs and .meta) is so completed by the next run that creates the key.
LONG c67_checked, c67_sets_done, c67_parts_written, c67_incomplete, c67_failed;  // this run
void complete_c67_request(u64 key, Wrapper* pipeline, const PipelineInfo* info, const void* code, u64 code_bytes, const void* meta,
                          u64 meta_bytes, const std::vector<unsigned char>& root_blob)
{
    static SRWLOCK lock = SRWLOCK_INIT;
    static std::unordered_set<u64> keys;  // handled this run
    wchar_t path[MAX_PATH];
    if (!cache_path(path, L"req_c67_", key, L".done")) return;
    AcquireSRWLockExclusive(&lock);
    const bool fresh = keys.insert(key).second;
    ReleaseSRWLockExclusive(&lock);
    if (!fresh || GetFileAttributesW(path) != INVALID_FILE_ATTRIBUTES) return;
    InterlockedIncrement(&c67_checked);
    std::vector<unsigned char> regs;
    AcquireSRWLockShared(&compute_lock);
    if (const auto found = compute_regs().find(pipeline); found != compute_regs().end()) regs = found->second;
    ReleaseSRWLockShared(&compute_lock);
    const bool have_meta = meta_bytes && meta_bytes <= 0x10000 && x2w_local::readable(meta, meta_bytes);
    struct Part { const wchar_t* ext; const void* data; std::size_t bytes; bool available; };
    const Part parts[] = {
        {L".cs", code, static_cast<std::size_t>(code_bytes), true},
        {L".meta", meta, have_meta ? static_cast<std::size_t>(meta_bytes) : 0, have_meta},
        {L".root", root_blob.data(), root_blob.size(), true},
        {L".desc", info->desc.data(), info->desc.size(), true},
        {L".regs", regs.data(), regs.size(), !regs.empty()},
    };
    int present = 0, written = 0;
    bool failed = false;
    for (const Part& part : parts) {
        WIN32_FILE_ATTRIBUTE_DATA a{};
        if (!cache_path(path, L"req_c67_", key, part.ext)) return;
        const bool on_disk = GetFileAttributesExW(path, GetFileExInfoStandard, &a);
        const u64 size = (u64{a.nFileSizeHigh} << 32) | a.nFileSizeLow;
        if (on_disk && (!part.available || size == part.bytes)) {
            ++present;
            continue;
        }
        if (!part.available) continue;
        if (regblob::write_whole(path, part.data, part.bytes)) ++present, ++written;
        else failed = true;
    }
    InterlockedAdd(&c67_parts_written, written);
    const bool whole = present == static_cast<int>(std::size(parts));
    static const char marker[] = "ok\n";
    if (whole && cache_path(path, L"req_c67_", key, L".done") && regblob::write_whole(path, marker, 3)) InterlockedIncrement(&c67_sets_done);
    else InterlockedIncrement(failed ? &c67_failed : &c67_incomplete);  // retried by the next run
    static LONG logged;
    if ((written || !whole) && first_calls(logged, 6))
        log("x2w-cx request %016llx: %d of 5 parts on disk, %d written now%s", key, present, written,
            whole ? ", .done written" : failed ? ", a write failed (no .done)" : ", no .regs or .meta yet (no .done)");
}
// The compute key of the pipeline bound on list self (also used by indirect dispatches).
u64 compute_key_for(Wrapper* pipeline);
bool prewarm_enabled();      // d3d12_compute.inl
bool prewarm_compute(u64 key, bool front = false);
u64 compute_key_of(Wrapper* self)
{
    Pass& pass = pass_of(self);
    u64 key = ~0ull;  // no compute pipeline bound
    AcquireSRWLockExclusive(&compute_lock);
    if (pass.pipeline) {
        const auto found = compute_keys().find(pass.pipeline);
        key = found != compute_keys().end() ? found->second : ~1ull;  // ~1: no compute key known for it
    }
    ReleaseSRWLockExclusive(&compute_lock);
    return key == ~1ull ? compute_key_for(pass.pipeline) : key;
}
// The compute key of a pipeline, derived from its creation description and remembered in compute_keys()
// (also at creation, for the background prewarm in d3d12_compute.inl).
u64 compute_key_for(Wrapper* pipeline)
{
    u64 key = ~1ull;
    {
        // Device slot 67: a console compute pipeline. Its description has the layout of slot 57:
        // [0] root signature, [5]/[6] machine code address and length, [7]/[8] the stage's metadata.
        // The key is fnv(root blob, fnv(code)), tagged with 0x67 in the top byte; the code, metadata
        // and root blob are written to the shader cache as req_c67_<key>.{cs,meta,root,desc,regs}, then .done
        // (complete_c67_request).
        const PipelineInfo* info = pipeline_info(pipeline);
        if (info && info->path == 67 && info->desc.size() >= 0x48) {
            u64 words[9];
            std::memcpy(words, info->desc.data(), sizeof(words));
            std::vector<unsigned char> root_blob;
            if (Wrapper* root = as_wrapper(reinterpret_cast<const void*>(words[0]))) {
                AcquireSRWLockShared(&note_lock);
                const auto found = root_blobs().find(root);
                if (found != root_blobs().end()) root_blob = found->second;
                ReleaseSRWLockShared(&note_lock);
            }
            const auto* code = reinterpret_cast<const void*>(words[5]);
            const auto* meta = reinterpret_cast<const void*>(words[7]);
            if (words[6] && words[6] <= (1u << 20) && x2w_local::readable(code, words[6])) {
                key = (fnv_bytes(root_blob.data(), root_blob.size(), fnv_bytes(code, words[6])) & 0x00ffffffffffffffull) | (0x67ull << 56);
                complete_c67_request(key, pipeline, info, code, words[6], meta, words[8], root_blob);
            } else {
                key = 0x67ull << 56;  // code unreadable
            }
        } else if (info) {
            key = (0xeeull << 56) | static_cast<u64>(info->path);  // another creation path
        }
        AcquireSRWLockExclusive(&compute_lock);
        compute_keys()[pipeline] = key;
        ReleaseSRWLockExclusive(&compute_lock);
    }
    return key;
}
// List slot 14, Dispatch(x, y, z). LOCAL instrumentation: counts dispatches per compute pipeline key.
void __stdcall list_dispatch(Wrapper* self, UINT x, UINT y, UINT z)
{
    const u64 key = compute_key_of(self);
    InterlockedIncrement(&frame_dispatches);
    const bool executed = execute_dispatch(self, x, y, z, key);
    if (trace_frame_active) log("x2w-frame dispatch %016llx (%u, %u, %u)%s", key, x, y, z, executed ? " executed" : "");
}
void report_dispatches()
{
    std::vector<std::pair<u64, DispatchCount>> all;
    AcquireSRWLockExclusive(&compute_lock);
    LONG o[co_count];
    for (const auto& entry : dispatch_counts()) all.push_back(entry);
    dispatch_counts().clear();
    for (int i = 0; i < co_count; ++i) o[i] = compute_outcomes[i], compute_outcomes[i] = 0;  // the keys' window
    ReleaseSRWLockExclusive(&compute_lock);
    std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) { return a.second.calls > b.second.calls; });
    // LOCAL: every key is listed; a full line continues on another line with the same prefix, and a closing
    // line gives the key and line counts so a reader can tell the list is complete.
    static const char prefix[] = "x2w-render dispatches per compute key (300 frames):";
    // log() cuts a line at 1533 characters (its 1536-byte buffer): stay below it, or the tail keys are lost.
    char line[1400];
    int n = std::snprintf(line, sizeof(line), "%s", prefix), lines = 0;
    for (const auto& entry : all) {
        if (n > static_cast<int>(sizeof(line)) - 80) {
            log("%s", line);
            ++lines;
            n = std::snprintf(line, sizeof(line), "%s", prefix);
        }
        wchar_t path[MAX_PATH];
        const bool signed_ready = cache_path(path, L"", entry.first, L".cs.dxbc") && GetFileAttributesW(path) != INVALID_FILE_ATTRIBUTES;
        n += std::snprintf(line + n, sizeof(line) - n, " %016llx%s:%lld(%u,%u,%u)", entry.first, signed_ready ? "*" : "",
                           entry.second.calls, entry.second.x, entry.second.y, entry.second.z);
    }
    log("%s", line);
    ++lines;
    log("x2w-render dispatch keys (300 frames): %zu keys in %d lines", all.size(), lines);
    log("x2w-compute dispatches (300 frames): executed %ld skipped %ld; slot-62 desktop pipeline executed %ld skipped %ld "
        "(table argument %ld, unmapped root address %ld); translated executed %ld skipped %ld (X2W_CX_OFF %ld, not a slot-67 key %ld, "
        "no translation %ld, heap or vinfo %ld); not run %ld (no list or pipeline, or X2W_COMPUTE_OFF); X2W_CX_SKIP skipped %ld; raw-address executed %ld; translated skipped, entry value unresolved %ld; translated skipped while compiling (X2W_CX_ASYNC) %ld",
        o[co_c62_run] + o[co_c67_run] + o[co_rawaddr_run], o[co_c62_table_arg] + o[co_c62_unmapped] + o[co_c67_off] + o[co_c67_other_key] + o[co_c67_untranslated] + o[co_c67_heap] + o[co_cx_skip] + o[co_c67_unresolved] + o[co_c67_compiling],
        o[co_c62_run], o[co_c62_table_arg] + o[co_c62_unmapped], o[co_c62_table_arg], o[co_c62_unmapped], o[co_c67_run],
        o[co_c67_off] + o[co_c67_other_key] + o[co_c67_untranslated] + o[co_c67_heap], o[co_c67_off], o[co_c67_other_key], o[co_c67_untranslated],
        o[co_c67_heap], o[co_not_run], o[co_cx_skip], o[co_rawaddr_run], o[co_c67_unresolved], o[co_c67_compiling]);
    if (c67_checked)
        log("x2w-cx requests (this run): %ld keys without .done checked, %ld sets completed (.done written), %ld parts written, "
            "%ld incomplete (no .meta or .regs at creation), %ld with a failed write; incomplete ones are retried next run",
            c67_checked, c67_sets_done, c67_parts_written, c67_incomplete, c67_failed);
}
#include "startup_stubs/mesh86_cache.inl"
#include "startup_stubs/d3d12_compute_binding_trace.inl"
#include "startup_stubs/list_copy_texture.inl"

// Queue slot 10, desktop ExecuteCommandLists(count, lists).
void __stdcall queue_execute(Wrapper* self, UINT count, void* const* lists)
{
    X2W_ZONE("ExecuteCommandLists");
    static LONG calls;
    drain_retired_resources();
    if (render_mode()) {
        // A closed list stays submittable until its next Reset, as on the desktop.
        std::vector<ID3D12CommandList*> ready;
        std::vector<Wrapper*> submitted;
        // Pin the closed recording and its allocator through submission bookkeeping.
        lock_draw(0 + __LINE__);  // LOCAL stall: X2W_DRAW_LOCK_TIMES
        AcquireSRWLockExclusive(&allocator_lock);
        for (UINT i = 0; i < count; ++i) {
            Wrapper* w = as_wrapper(lists[i]);
            if (w && w->own().kind == list_kind && w->own().list_ready) {
                ready.push_back(static_cast<ID3D12CommandList*>(w->own().real));
                submitted.push_back(w);
            }
        }
        auto* queue = static_cast<ID3D12CommandQueue*>(self->own().real);
        bool split = false;
        const auto submit = [&](const auto& native_lists, const auto& wrappers) {
            const UINT ready_count = static_cast<UINT>(native_lists.size());
            if (range_watch_count()) range_watch_sample("submit");  // LOCAL
            auto title_reads = prepare_texture_readbacks(queue, native_lists);
            red_producers().execute(queue, native_lists, [&] {
                if (x2w_gpu::gpu_abi_on()) gpu_list_submit(queue, ready_count, wrappers.data());
                if (draw_dump_frame()) {
                    const HRESULT hr = draw_dumps().execute(queue, ready_count, native_lists.data());
                    if (FAILED(hr)) log("x2w-draw readback fence failed 0x%08lx", static_cast<unsigned long>(hr));
                } else {
                    queue->ExecuteCommandLists(ready_count, native_lists.data());
                }
            });
            submit_texture_readbacks(queue, std::move(title_reads));
            cull_capture_submitted(queue, native_lists.data(), ready_count);  // LOCAL: X2W_CULL_INPUT_CAPTURE
            if (x2w_gpu::gpu_abi_on()) gpu_list_submitted(queue, ready_count, wrappers.data());
            retire_address_uploads(queue, native_lists);
            retire_draw_sampler_heaps(queue, native_lists);
            note_submits(queue, wrappers);
            retire_note_submission(queue, wrappers);
            for (Wrapper* w : wrappers) commit_list_gpu_writes(w);
            InterlockedIncrement(&render_executes);
            return trace_mesh_xmem_submission(queue, native_lists, split);
        };
        if (mesh_xmem_tracking()) {
            const std::lock_guard lock(mesh_xmem_mutex);
            reap_mesh_xmem_flights();
            split = std::any_of(ready.begin(), ready.end(), mesh_xmem_needs_submission);
        }
        if (split) {
            // Submit each recording in guest order so a writer's publication
            // belongs to its own completion point, even within a guest batch.
            std::vector<ID3D12CommandList*> native_list(1);
            std::vector<Wrapper*> wrapper(1);
            for (std::size_t i = 0; i < ready.size(); ++i) {
                native_list[0] = ready[i];
                wrapper[0] = submitted[i];
                execute_mesh_xmem_recording(queue, ready[i], [&] { return submit(native_list, wrapper); });
            }
        } else if (!ready.empty()) {
            submit(ready, submitted);
        }
        ReleaseSRWLockExclusive(&allocator_lock);
        if (!ready.empty()) {
            // LOCAL: still under draw_lock as before, but after allocator_lock: the memcut reaper and the upload
            // ring take it (shared) themselves.
            memcut_reap();  // LOCAL memcut (X2W_MEMCUT_RELEASE)
            ring_submitted(submitted);  // LOCAL memcut (upload ring)
            gpu_time::submitted(static_cast<ID3D12CommandQueue*>(self->own().real), submitted);  // LOCAL: X2W_CX_GPU_TIME
            static const bool srv_report_submits = x2w_local::env_on("X2W_DRAW_SRV_STATS_SUBMITS");  // LOCAL stall: fixtures only
            if (srv_report_submits) report_srv_copies();
        }
        ReleaseSRWLockExclusive(&draw_lock);
        check_device("CommandQueue execute", 35);
        return;
    }
    if (list_noop()) {
        if (first_calls(calls)) log("x2w-d3d12 CommandQueue::ExecuteCommandLists count=%u -> not submitted (noop lists)", count);
        return;
    }
    // No hold or flight is recorded here: the copy handlers that create holds exist only in render mode.
    std::vector<ID3D12CommandList*> real(count);
    for (UINT i = 0; i < count; ++i) real[i] = reinterpret_cast<ID3D12CommandList*>(unwrap(reinterpret_cast<u64>(lists[i])));
    static_cast<ID3D12CommandQueue*>(self->own().real)->ExecuteCommandLists(count, real.data());
    if (first_calls(calls)) log("x2w-d3d12 CommandQueue::ExecuteCommandLists count=%u", count);
}

// Fence slot 10, desktop Signal(value) (also used by the DirectStorage stand-in's EnqueueSignal).
HRESULT __stdcall fence_signal(Wrapper* self, UINT64 value)
{
    const HRESULT hr = static_cast<ID3D12Fence*>(self->own().real)->Signal(value);
    fence_reached(self, value);
    return hr;
}
// Fence slot 8, desktop GetCompletedValue().
UINT64 __stdcall fence_completed_value(Wrapper* self) { return static_cast<UINT64>(self->own().completed); }
// Fence slot 12 (Xbox order): (value, ptr, flag) -> address of the completed value, which the title
// reads and waits on directly; all ones there means device removed.
volatile LONG64* __stdcall fence_completed_address(Wrapper* self, u64 value, u64 r8, u64 r9)
{
    static LONG calls;
    if (first_calls(calls, 2))
        log("x2w-d3d12 Fence::slot12(completed value address) value=0x%llx r8=0x%llx r9=0x%llx -> %p (now 0x%llx)", value, r8,
            r9, static_cast<void*>(const_cast<LONG64*>(&self->own().completed)), static_cast<unsigned long long>(self->own().completed));
    return &self->own().completed;
}

// Queue slot 18 (Xbox order): (UINT64* frequency) -> HRESULT; the title divides by the value.
HRESULT __stdcall queue_timestamp_frequency(Wrapper* self, UINT64* frequency)
{
    const HRESULT hr = static_cast<ID3D12CommandQueue*>(self->own().real)->GetTimestampFrequency(frequency);
    static LONG calls;
    if (first_calls(calls, 2))
        log("x2w-d3d12 CommandQueue::slot18(GetTimestampFrequency) -> 0x%08lx %llu", static_cast<unsigned long>(hr),
            frequency ? *frequency : 0ull);
    return hr;
}

// The page grants, section offsets and destination views are owned by XMem.
// Hold the installed provider for this batch; a queue can predate xmem.dll.
HRESULT WINAPI queue_page_mappings(Wrapper*, UINT count, const X2WPageBatch* batches,
                                   UINT64 pool, UINT pool_pages, UINT flags)
{
    HMODULE xmem = nullptr;
    if (!GetModuleHandleExW(0, L"xmem.dll", &xmem)) return HRESULT_FROM_WIN32(GetLastError());
    const auto map = reinterpret_cast<decltype(&X2WMapPageBatch)>(GetProcAddress(xmem, "X2WMapPageBatch"));
    X2WPageBatchStats stats{};
    const HRESULT hr = map ? map(count, batches, pool, pool_pages, flags, &stats)
                           : HRESULT_FROM_WIN32(ERROR_PROC_NOT_FOUND);
    FreeLibrary(xmem);
    if (SUCCEEDED(hr) && stats.pagesChanged) x2w_local::invalidate_probe_cache();
    if (FAILED(hr)) {
        static LONG failures;
        if (first_calls(failures, 2))
            log("x2w-d3d12 CommandQueue::slot31 count=%u poolPages=%u flags=0x%x -> 0x%08lx",
                count, pool_pages, flags, static_cast<unsigned long>(hr));
    }
    return hr;
}

// Queue slots whose Xbox meaning is unknown, dispatched by argument shape:
// - (fence wrapper, value): Signal on the desktop queue. Slot 17 (Wait) has the same shape and is
//   handled by queue_wait below; on other slots a wait would complete the fence early.
// - (count, array of command-list wrappers): ExecuteCommandLists (not submitted while lists are no-ops).
// Anything else logs its first call and returns 0. Each shape logs the slot it arrived on.
template <int N>
__declspec(noinline) u64 queue_dynamic(Wrapper* self, u64 rdx, u64 r8, u64 r9)
{
    Timed timed(time_queue);
    static LONG calls;
    const bool log_it = first_calls(calls, 2);
    if (Wrapper* fence = as_wrapper(reinterpret_cast<void*>(rdx)); fence && fence->own().kind == fence_kind) {
        const HRESULT hr = static_cast<ID3D12CommandQueue*>(self->own().real)->Signal(static_cast<ID3D12Fence*>(fence->own().real), r8);
        if (SUCCEEDED(hr)) {
            if (render_mode() || InterlockedCompareExchange(&self->own().queue_wait_seen, 0, 0))
                complete_when_reached(fence, r8);
            else
                fence_reached(fence, r8);
        }
        if (log_it)
            log("x2w-d3d12 CommandQueue::slot%d(fence=%p, value=0x%llx) -> Signal 0x%08lx", N, static_cast<void*>(fence), r8,
                static_cast<unsigned long>(hr));
        return static_cast<u64>(hr);
    }
    if (rdx >= 1 && rdx <= 64 && r8 >= 0x10000 && x2w_local::readable(reinterpret_cast<void*>(r8), rdx * sizeof(void*))) {
        auto* const* lists = reinterpret_cast<void* const*>(r8);
        bool all_lists = true;
        for (u64 i = 0; i < rdx; ++i) {
            Wrapper* w = as_wrapper(lists[i]);
            all_lists &= w && w->own().kind == list_kind;
        }
        if (all_lists) {
            if (log_it) log("x2w-d3d12 CommandQueue::slot%d(count=%llu, lists) -> ExecuteCommandLists", N, rdx);
            queue_execute(self, static_cast<UINT>(rdx), lists);
            return 0;
        }
    }
    if (create_shaped(rdx, r8, r9, 0, 0, 0, 0)) {
        char name[64];
        std::snprintf(name, sizeof(name), "CommandQueue::slot%d(passes an IID)", N);
        x2w_trap::fire(X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, name, N & 0xFF, reinterpret_cast<u64>(self), rdx, r8, r9,
                       _ReturnAddress());
    }
    if (noop_census_enabled()) {
        const u64 args[8] = {reinterpret_cast<u64>(self), rdx, r8, r9, 0, 0, 0, 0};
        noop_census_note(queue_kind, N, args, reinterpret_cast<u64>(_ReturnAddress()) - reinterpret_cast<u64>(GetModuleHandleW(nullptr)));
    }
    if (log_it) {
        char name[64];
        std::snprintf(name, sizeof(name), "CommandQueue::slot%d", N);
        x2w_trap::record("x2w-noop", X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, name, N & 0xFF, reinterpret_cast<u64>(self), rdx, r8,
                         r9, _ReturnAddress());
    }
    return 0;
}

// Queue slot 17 (Xbox order): (fence, value) -> HRESULT, Wait. The console queue has two methods
// before the desktop's Signal and Wait (desktop slots 14 and 15), so Signal is slot 16 and Wait is
// slot 17; slot 18 (GetTimestampFrequency) and slot 32 (present) follow the same layout. The desktop
// queue waits on the GPU for the fence value. Any other argument shape keeps the shape dispatch.
u64 __stdcall queue_wait(Wrapper* self, u64 rdx, u64 r8, u64 r9)
{
    Wrapper* fence = as_wrapper(reinterpret_cast<void*>(rdx));
    if (!fence || fence->own().kind != fence_kind) return queue_dynamic<17>(self, rdx, r8, r9);
    InterlockedExchange(&self->own().queue_wait_seen, 1);
    const HRESULT hr = static_cast<ID3D12CommandQueue*>(self->own().real)->Wait(static_cast<ID3D12Fence*>(fence->own().real), r8);
    static LONG calls;
    if (first_calls(calls, 2))
        log("x2w-d3d12 CommandQueue::slot17(fence=%p, value=0x%llx) -> Wait 0x%08lx", static_cast<void*>(fence), r8,
            static_cast<unsigned long>(hr));
    return static_cast<u64>(hr);
}

template <std::size_t... I>
void fill_queue_dynamic(void** table, std::index_sequence<I...>)
{
    ((I >= 8 ? (table[I] = reinterpret_cast<void*>(&queue_dynamic<static_cast<int>(I)>)) : nullptr), ...);
}

// Queue GetDesc(), returned by value (desktop slot 18; Xbox slot unknown, not installed).
void* __stdcall queue_get_desc(void* a, void* b)
{
    void* result = nullptr;
    Wrapper* self = receiver(a, b, &result, "CommandQueue::GetDesc(no receiver)");
    *static_cast<D3D12_COMMAND_QUEUE_DESC*>(result) = static_cast<ID3D12CommandQueue*>(self->own().real)->GetDesc();
    return result;
}

// Resource slot 8, desktop Map(subresource, readRange, data). On the console a resource's memory is
// title-visible: for a resource placed at a title address or backed by a hook allocation, the mapped
// pointer is that memory (subresource offsets inside it are not modeled). Others map on desktop.
HRESULT __stdcall resource_map(Wrapper* self, UINT subresource, const D3D12_RANGE* range, void** data)
{
    HRESULT hr = S_OK;
    void* mapped = self->own().title_memory;
    if (!mapped) hr = static_cast<ID3D12Resource*>(self->own().real)->Map(subresource, range, &mapped);
    if (data) *data = SUCCEEDED(hr) ? mapped : nullptr;
    if (FAILED(hr)) check_device("Resource::Map", 8);
    static LONG calls, failures;
    if (first_calls(calls, 4) || (FAILED(hr) && first_calls(failures, 8)))
        log("x2w-d3d12 Resource::Map subresource=%u title_memory=%p -> 0x%08lx %p", subresource, self->own().title_memory,
            static_cast<unsigned long>(hr), mapped);
    return hr;
}
void __stdcall resource_unmap(Wrapper* self, UINT subresource, const D3D12_RANGE* range)
{
    if (!self->own().title_memory) static_cast<ID3D12Resource*>(self->own().real)->Unmap(subresource, range);
}

// Resource slot 11, desktop GetGPUVirtualAddress() (called with no arguments). On the console a
// resource placed at an address has that address as its GPU address, so the title gets its own
// address back; resources without title memory report the desktop address (0 for textures).
UINT64 __stdcall resource_gpu_address(Wrapper* self)
{
    const UINT64 desktop = static_cast<ID3D12Resource*>(self->own().real)->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER
        ? static_cast<ID3D12Resource*>(self->own().real)->GetGPUVirtualAddress() : 0;
    const UINT64 result = self->own().title_memory ? reinterpret_cast<UINT64>(self->own().title_memory) : desktop;
    static LONG calls;
    if (first_calls(calls, 4))
        log("x2w-d3d12 Resource::GetGPUVirtualAddress title_memory=%p desktop=0x%llx -> 0x%llx", self->own().title_memory, desktop,
            result);
    return result;
}

// Resource slot 10, desktop GetDesc(), returned by value.
void* __stdcall resource_get_desc(void* a, void* b)
{
    void* result = nullptr;
    Wrapper* self = receiver(a, b, &result, "Resource::GetDesc(no receiver)");
    if (!self || !self->own().real || !result) return result;
    *static_cast<D3D12_RESOURCE_DESC*>(result) = static_cast<ID3D12Resource*>(self->own().real)->GetDesc();
    static LONG logged;
    if (first_calls(logged, 4)) {
        const auto& d = *static_cast<const D3D12_RESOURCE_DESC*>(result);
        log("x2w-d3d12 Resource::GetDesc res=%p dim=%d fmt=%d size=%llux%ux%u mips=%u layout=0x%x flags=0x%x",
            static_cast<void*>(self), d.Dimension, d.Format, static_cast<unsigned long long>(d.Width),
            d.Height, d.DepthOrArraySize, d.MipLevels, d.Layout, d.Flags);
    }
    return result;
}

// Heap slot 8, desktop GetDesc(), returned by value.
void* __stdcall heap_get_desc(void* a, void* b)
{
    void* result = nullptr;
    Wrapper* self = receiver(a, b, &result, "Heap::GetDesc(no receiver)");
    *static_cast<D3D12_HEAP_DESC*>(result) = static_cast<ID3D12Heap*>(self->own().real)->GetDesc();
    return result;
}

// Descriptor heap slots 8-10, desktop GetDesc / GetCPU/GPUDescriptorHandleForHeapStart, by value.
void* __stdcall descriptor_heap_get_desc(void* a, void* b)
{
    void* result = nullptr;
    Wrapper* self = receiver(a, b, &result, "DescriptorHeap::GetDesc(no receiver)");
    *static_cast<D3D12_DESCRIPTOR_HEAP_DESC*>(result) = static_cast<ID3D12DescriptorHeap*>(self->own().real)->GetDesc();
    return result;
}
// The title returns structures of at most 8 bytes in rax (this in rcx), larger ones through a hidden
template <class T, class Get>
std::uint64_t small_result(void* a, void* b, const char* method, Get get)
{
    Wrapper* w = as_wrapper(a);
    if (!w && a) {
        if (Wrapper* object = as_wrapper(b)) {
            const T value = get(object);
            *static_cast<T*>(a) = value;
            return reinterpret_cast<std::uint64_t>(a);
        }
    }
    if (!w) fail(method, a, b);
    const T value = get(w);
    std::uint64_t bits = 0;
    static_assert(sizeof(T) <= sizeof(bits));
    std::memcpy(&bits, &value, sizeof(T));
    return bits;
}
std::uint64_t __stdcall descriptor_heap_cpu_start(void* a, void* b)
{
    return small_result<D3D12_CPU_DESCRIPTOR_HANDLE>(a, b, "DescriptorHeap::GetCPUDescriptorHandleForHeapStart(no receiver)",
        [](Wrapper* w) {
            if (w->own().title_cpu) return D3D12_CPU_DESCRIPTOR_HANDLE{static_cast<SIZE_T>(w->own().title_cpu)};
            return static_cast<ID3D12DescriptorHeap*>(w->own().real)->GetCPUDescriptorHandleForHeapStart();
        });
}

// Device slot 14, desktop CreateDescriptorHeap(desc, riid, ppv). The title's descriptor is longer:
// +0x10 a title CPU address for the descriptors (or null) and +0x18 their size in bytes. With an
// address, GetCPUDescriptorHandleForHeapStart must return it (the title asserts this).
HRESULT __stdcall device_create_descriptor_heap(Wrapper* self, const D3D12_DESCRIPTOR_HEAP_DESC* desc, const GUID* iid,
                                                void** out)
{
    Timed timed(time_heap);
    char text[48];
    guid_text(iid, text, sizeof(text));
    if (!out || !desc || !iid) return E_POINTER;
    *out = nullptr;
    const auto* extra = reinterpret_cast<const std::uint64_t*>(desc + 1);
    ID3D12DescriptorHeap* heap = nullptr;
    HRESULT hr = kind_for(*iid) == descriptor_heap_kind
        ? device_of(self)->CreateDescriptorHeap(desc, IID_PPV_ARGS(&heap)) : E_NOINTERFACE;
    // A console view heap can be larger than the driver accepts. Measure the accepted maximum, create
    // the desktop heap at it, and keep the title's full-size descriptor memory. Desktop view writes
    // and copies past the physical prefix use a companion CPU-only heap.
    // The refusal code differs between drivers (E_INVALIDARG on a GTX 1660 SUPER, E_OUTOFMEMORY on an
    // RTX 3060). Preserve errors unrelated to capacity, including unsupported interface IDs.
    D3D12_DESCRIPTOR_HEAP_DESC clamped = *desc;
    bool did_clamp = false;
    if ((hr == E_INVALIDARG || hr == E_OUTOFMEMORY) && (desc->Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)
        && desc->Type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        && desc->NumDescriptors > view_heap_probe_floor) {
        const std::uint64_t limit = accepted_view_heap_descriptors(device_of(self), desc->NodeMask);
        if (limit && limit < desc->NumDescriptors) {
            clamped.NumDescriptors = static_cast<UINT>(limit);
            did_clamp = true;
            hr = device_of(self)->CreateDescriptorHeap(&clamped, IID_PPV_ARGS(&heap));
            static LONG shortcuts;
            if (SUCCEEDED(hr) && first_calls(shortcuts, 1))
                log("x2w-d3d12 Device::CreateDescriptorHeap: shortcut - title asked for %u shader-visible "
                    "view descriptors, this driver accepts %u; the rest stay in title memory",
                    desc->NumDescriptors, clamped.NumDescriptors);
        }
    }
    const D3D12_DESCRIPTOR_HEAP_DESC* request = did_clamp ? &clamped : desc;
    const std::uint64_t desktop = heap ? heap->GetCPUDescriptorHandleForHeapStart().ptr : 0;
    static LONG calls;
    if (first_calls(calls, 16))
        log("x2w-d3d12 Device::CreateDescriptorHeap type=%d count=%u flags=0x%x node=0x%x title=0x%llx bytes=0x%llx riid=%s "
            "-> 0x%08lx desktop=0x%llx", desc->Type, desc->NumDescriptors, desc->Flags, desc->NodeMask, extra[0], extra[1],
            text, static_cast<unsigned long>(hr), desktop);
    if (FAILED(hr)) return hr;
    x2w_startup::ComPtr<ID3D12DescriptorHeap> tail;
    if (did_clamp) {
        D3D12_DESCRIPTOR_HEAP_DESC tail_desc = *desc;
        tail_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        tail_desc.NumDescriptors -= clamped.NumDescriptors;
        hr = device_of(self)->CreateDescriptorHeap(&tail_desc, IID_PPV_ARGS(&tail));
        // Retain the physical CPU tail for the lifetime of the original desktop heap.
        static constexpr GUID tail_lifetime{0x1aa68383,0xf1e9,0x4264,{0xa7,0x05,0x1f,0xf9,0x69,0xa8,0x03,0xac}};
        if (SUCCEEDED(hr)) hr = heap->SetPrivateDataInterface(tail_lifetime, tail.Get());
        if (FAILED(hr)) { heap->Release(); return hr; }
    }
    const UINT desktop_stride = device_of(self)->GetDescriptorHandleIncrementSize(desc->Type);
    const bool raw_samplers = desc->Type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER &&
        extra[1] == u64{desc->NumDescriptors} * x2w_startup::title_sampler_stride;
    // Translated shader ABI: the title's views are 32 and its samplers 16 bytes apart (its slot 15 answers), whatever
    // the desktop increments are (WARP: 64 and 64).
    const bool gpu_views = x2w_gpu::gpu_abi_on() && desc->Type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    const bool gpu_samplers = x2w_gpu::gpu_abi_on() && desc->Type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    const UINT title_stride = raw_samplers || gpu_samplers ? x2w_startup::title_sampler_stride
        : gpu_views ? x2w_gpu::abi::view_increment : desktop_stride;
    const u64 bytes = u64{desc->NumDescriptors} * title_stride;
    // Some callers pass the 16-byte desktop descriptor; the words after it are then unrelated stack
    // data. The extension is taken only when it holds a pointer and exactly the heap's size.
    std::uint64_t title_memory = extra[0] >= 0x10000 && extra[1] == bytes ? extra[0] : 0;
    // The title copies descriptors directly between heaps. Allocate CPU-visible backing when a heap lacks an
    // addressable desktop start so the view IDs written by note_view survive byte copies. The desktop heap is still
    // populated. Unplaced sampler heaps, including CPU-only staging heaps, use 16-byte title records.
    if (!x2w_startup::ensure_title_heap_memory(desc->Type, desktop, bytes, title_memory, [](u64 size) {
        return reinterpret_cast<u64>(VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    }, gpu_samplers)) {
        heap->Release();
        return E_OUTOFMEMORY;
    }
    Wrapper* w = wrap(heap, descriptor_heap_kind);
    if (title_memory) {
        w->own().title_cpu = title_memory;
        AcquireSRWLockExclusive(&descriptor_lock);
        descriptor_ranges.push_back({title_memory, bytes, desktop, title_stride, desktop_stride,
                                     static_cast<std::uint64_t>(request->NumDescriptors),
                                     tail ? tail->GetCPUDescriptorHandleForHeapStart().ptr : 0});
        ReleaseSRWLockExclusive(&descriptor_lock);
    }
    const u64 sampler_cpu = raw_samplers ? title_memory : 0;
    note_heap(w, desc->Type, request->NumDescriptors, desc->NumDescriptors, desktop_stride,
              title_memory, desktop,
              (desc->Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) ? heap->GetGPUDescriptorHandleForHeapStart().ptr : 0,
              sampler_cpu);
    *out = w;
    return S_OK;
}

// Device view slots: forwarded with the destination CPU handle (argument D, 0-based after the
// object) translated from title to desktop.
// Desktop removes the device (DXGI_ERROR_INVALID_CALL) on view calls it rejects, so they are checked
// first and skipped with a log: a view type the desktop resource's flags do not allow (the console
// makes unordered-access views of depth textures), and constant-buffer views, whose BufferLocation is
// a title GPU address. Console format values in the view description are mapped like resource formats.
template <int N, int D>
__declspec(noinline) u64 view_slot(Wrapper* self, u64 a, u64 b, u64 c, u64 d)
{
    Timed timed(time_view);
    // UAV, RTV and DSV views keep the resource's old lifetime; slot-18 SRVs stay in the retire class as long as
    // their descriptor slots are tracked by shadow_slot and translated draws hold bound resources with retire_hold.
    if constexpr (N >= 19 && N <= 21) retire_leave_class(reinterpret_cast<const void*>(a), RetireLeft::view);
    if constexpr (N == 19) retire_leave_class(reinterpret_cast<const void*>(b), RetireLeft::view);
    u64 args[4] = {unwrap(a), unwrap(b), unwrap(c), unwrap(d)};
    const u64 title_dest = args[D];
    args[D] = desktop_cpu_handle(args[D]);
    if (!args[D]) {
        if (noop_census_enabled()) noop_census_dropped_view(N, title_dest);
        if constexpr (N == 18) retire_leave_class(reinterpret_cast<const void*>(a), RetireLeft::view);
        return 0;
    }
    const void* adapted_record = nullptr;
    const auto record_view = [&](bool materialized) {
        const u64 raw[4] = {a, b, c, d};
        const int described = N == 19 ? 2 : (N == 17 || N == 22) ? 0 : 1;
        const std::size_t described_bytes = N == 17 ? 16 : N == 22 ? 52 : N == 18 ? 40 : N == 19 ? sizeof(D3D12_UNORDERED_ACCESS_VIEW_DESC) : N == 20 ? 20 : 16;
        note_view(N, (N >= 18 && N <= 21) ? raw[0] : 0, N == 19 ? raw[1] : 0, adapted_record ? adapted_record : reinterpret_cast<const void*>(raw[described]),
                  described_bytes, title_dest, args[D], materialized);
    };
    const char* skip = nullptr;
    if (N == 17) skip = "constant buffer view (title GPU address)";
    if (N >= 18 && N <= 21 && args[0]) {
        const UINT flags = reinterpret_cast<ID3D12Resource*>(args[0])->GetDesc().Flags;
        if (N == 18 && (flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)) skip = "resource denies shader resource views";
        if (N == 19 && !(flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) skip = "resource lacks unordered access";
        if (N == 20 && !(flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)) skip = "resource lacks render target";
        if (N == 21 && !(flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)) skip = "resource lacks depth stencil";
    }
    if (skip) {
        record_view(false);
        static LONG skipped;
        if (first_calls(skipped, 6)) log("x2w-d3d12 Device::slot%d view skipped: %s", N, skip);
        return 0;
    }
    // The view descriptions of slots 18-21 start with a DXGI_FORMAT.
    alignas(D3D12_SHADER_RESOURCE_VIEW_DESC) unsigned char desc_copy[64];
    const int desc_index = N == 19 ? 2 : 1;
    constexpr std::size_t view_bytes = N == 18 ? sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC) :
        N == 19 ? sizeof(D3D12_UNORDERED_ACCESS_VIEW_DESC) :
        N == 20 ? sizeof(D3D12_RENDER_TARGET_VIEW_DESC) : sizeof(D3D12_DEPTH_STENCIL_VIEW_DESC);
    if constexpr (N >= 18 && N <= 20) {
        const auto* owner = as_wrapper(reinterpret_cast<const void*>(a));
        if (owner && owner->own().srgb_uav_format) {
            const auto original = static_cast<DXGI_FORMAT>(owner->own().srgb_uav_format);
            const auto rd = reinterpret_cast<ID3D12Resource*>(args[0])->GetDesc();
            if (!args[desc_index]) {
                if constexpr (N == 18) { const auto v = x2w_srgb::srv(rd, original); std::memcpy(desc_copy, &v, sizeof(v)); }
                if constexpr (N == 19) { const auto v = x2w_srgb::uav(rd, original); std::memcpy(desc_copy, &v, sizeof(v)); }
                if constexpr (N == 20) { const auto v = x2w_srgb::rtv(rd, original); std::memcpy(desc_copy, &v, sizeof(v)); }
                adapted_record = desc_copy;
            } else if (x2w_local::readable(reinterpret_cast<const void*>(args[desc_index]), view_bytes)) {
                std::memcpy(desc_copy, reinterpret_cast<const void*>(args[desc_index]), view_bytes);
                DXGI_FORMAT format;
                std::memcpy(&format, desc_copy, sizeof(format));
                if (format == DXGI_FORMAT_UNKNOWN) format = original;
                if constexpr (N == 19) format = x2w_srgb::linear(format);
                std::memcpy(desc_copy, &format, sizeof(format));
                adapted_record = desc_copy;
            }
            if (adapted_record) args[desc_index] = reinterpret_cast<u64>(adapted_record);
        }
    }
    if (N >= 18 && N <= 21 && args[desc_index] && x2w_local::readable(reinterpret_cast<const void*>(args[desc_index]), view_bytes)) {
        if (!adapted_record) std::memcpy(desc_copy, reinterpret_cast<const void*>(args[desc_index]), view_bytes);
        DXGI_FORMAT format;
        std::memcpy(&format, desc_copy, sizeof(format));
        DXGI_FORMAT mapped = desktop_format(format);
        if (args[0] && x2w_startup::packed_target_format(mapped)) {
            // A packed view follows a stand-in texture and is refused on any other texture of another format.
            auto* const resource = reinterpret_cast<ID3D12Resource*>(args[0]);
            const auto actual = resource->GetDesc().Format;
            const auto verdict = x2w_startup::packed_view(mapped, actual, x2w_startup::packed_stand_in_of(resource));
            if (verdict == x2w_startup::PackedView::refused) {
                record_view(false);
                static LONG refused;
                if (first_calls(refused, 6))
                    log("x2w-d3d12 Device::slot%d view skipped: packed format %d on a texture of format %d that does not stand in for it",
                        N, mapped, actual);
                return 0;
            }
            if (verdict == x2w_startup::PackedView::stand_in) mapped = actual;
        }
        if (mapped != format) {
            std::memcpy(desc_copy, &mapped, sizeof(mapped));
            args[desc_index] = reinterpret_cast<u64>(desc_copy);
        }
        if (N == 18) {
            D3D12_SHADER_RESOURCE_VIEW_DESC view;
            std::memcpy(&view, desc_copy, sizeof(view));
            const D3D12_SHADER_RESOURCE_VIEW_DESC before = view;
            fix_srv(view, reinterpret_cast<ID3D12Resource*>(args[0]));
            if (std::memcmp(&view, &before, sizeof(view)) != 0) {
                std::memcpy(desc_copy, &view, sizeof(view));
                args[desc_index] = reinterpret_cast<u64>(desc_copy);
                static LONG planes;
                if (first_calls(planes, 4)) log("x2w-d3d12 Device::slot18 view format %d: plane slice set for the depth-stencil plane", view.Format);
            }
        }
    }
    // A typeless depth texture (typeless_depth_family) needs a description; the title passed none.
    D3D12_SHADER_RESOURCE_VIEW_DESC default_srv{};
    D3D12_DEPTH_STENCIL_VIEW_DESC default_dsv{};
    if (N == 18 && args[0] && !args[1] && default_depth_srv(reinterpret_cast<ID3D12Resource*>(args[0]), default_srv))
        args[1] = reinterpret_cast<u64>(&default_srv);
    if (N == 21 && args[0] && !args[1] && default_depth_dsv(reinterpret_cast<ID3D12Resource*>(args[0]), default_dsv))
        args[1] = reinterpret_cast<u64>(&default_dsv);
    D3D12_UNORDERED_ACCESS_VIEW_DESC null_uav{};
    D3D12_RENDER_TARGET_VIEW_DESC null_rtv{};
    if constexpr (N == 19) {
        if (c && x2w_local::readable(reinterpret_cast<const void*>(c), sizeof(null_uav)) &&
            x2w_local::adapt_null_view(device_of(self), a != 0, b != 0,
                *reinterpret_cast<const D3D12_UNORDERED_ACCESS_VIEW_DESC*>(c), null_uav))
            args[2] = reinterpret_cast<u64>(&null_uav);
    }
    if constexpr (N == 20) {
        if (b && x2w_local::readable(reinterpret_cast<const void*>(b), sizeof(null_rtv)) &&
            x2w_local::adapt_null_view(device_of(self), a != 0,
                *reinterpret_cast<const D3D12_RENDER_TARGET_VIEW_DESC*>(b), null_rtv))
            args[1] = reinterpret_cast<u64>(&null_rtv);
    }
    bool trace_uav = false;
    if constexpr (N == 19) trace_uav = gpu_view_trace::watching(removal_device);
    const u64 result = method_of(self->own().real, N)(self->own().real, args[0], args[1], args[2], args[3], 0, 0, 0, 0);
    // A desktop view creation returns void, so the only failure it reports is a removed device. Only a created view
    // may keep the resource: on failure the slot is recorded unmaterialized, which releases the slot's previous view
    // reference and adds none.
    const HRESULT removed = removal_device ? removal_device->GetDeviceRemovedReason() : S_OK;
    if constexpr (N == 19) {
        if (trace_uav) gpu_view_trace::after("Device::slot19", "original UAV", removed,
            reinterpret_cast<ID3D12Resource*>(args[0]), reinterpret_cast<ID3D12Resource*>(args[1]),
            reinterpret_cast<const D3D12_UNORDERED_ACCESS_VIEW_DESC*>(args[2]), {static_cast<SIZE_T>(args[D])});
    }
    const bool created = SUCCEEDED(removed);
    if constexpr (N == 18 || N == 19) {
        if (!a) adapted_record = reinterpret_cast<const void*>(args[desc_index]);
    }
    record_view(created);
    if (!created) {
        static LONG failed;
        if (first_calls(failed, 6))
            log("x2w-d3d12 Device::slot%d view not created: device removed, reason 0x%08lx", N,
                static_cast<unsigned long>(removed));
    }
    if ((N == 20 || N == 21) && args[0] && created)
        note_view_target(args[D], reinterpret_cast<ID3D12Resource*>(args[0]), N);
    check_device("Device view", N);
    static LONG calls;
    if (first_calls(calls))
        log("x2w-d3d12 Device::slot%d view(0x%llx, 0x%llx, 0x%llx, 0x%llx) dest 0x%llx -> 0x%llx", N, a, b, c, d,
            D == 0 ? a : D == 1 ? b : D == 2 ? c : d, args[D]);
    return result;
}

void copy_lazy_native_srv(Wrapper*, u64 source, u64 destination);
// Copy through actual desktop heaps. A logical range may straddle the last
// shader-visible descriptor and the first CPU-only descriptor.
void copy_descriptor_run(Wrapper* self, UINT count, u64 dest, u64 src, D3D12_DESCRIPTOR_HEAP_TYPE type)
{
    const UINT increment = device_of(self)->GetDescriptorHandleIncrementSize(type);
    const UINT dest_stride = title_descriptor_stride(dest, increment);
    const UINT src_stride = title_descriptor_stride(src, increment);
    while (count) {
        UINT dest_room, src_room, dest_total, src_total;
        const auto d = desktop_cpu_handle(dest, &dest_room, &dest_total);
        const auto s = desktop_cpu_handle(src, &src_room, &src_total);
        if (count > dest_total || count > src_total || !d || !s) return;
        const UINT chunk = (std::min)(count, (std::min)(dest_room, src_room));
        device_of(self)->CopyDescriptorsSimple(chunk, {static_cast<SIZE_T>(d)}, {static_cast<SIZE_T>(s)}, type);
        note_copy(dest, src, d, s, chunk, increment);
        if (type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV && dest == d && src != s &&
            x2w_gpu::gpu_abi_on() && title_backed(src, u64{chunk} * src_stride)) {
            for (UINT i = 0; i < chunk; ++i) copy_lazy_native_srv(self, src + u64{i} * src_stride, dest + u64{i} * dest_stride);
        }
        dest += u64{chunk} * dest_stride;
        src += u64{chunk} * src_stride;
        count -= chunk;
    }
}

// Device slot 24, desktop CopyDescriptorsSimple(count, dest, src, type).
void __stdcall device_copy_descriptors_simple(Wrapper* self, UINT count, u64 dest, u64 src, D3D12_DESCRIPTOR_HEAP_TYPE type)
{
    Timed timed(time_copy);
    copy_descriptor_run(self, count, dest, src, type);
    check_device("Device CopyDescriptorsSimple", 24);
    static LONG calls;
    if (first_calls(calls)) log("x2w-d3d12 Device::CopyDescriptorsSimple count=%u dest=0x%llx src=0x%llx type=%d", count, dest, src, type);
}

// Device slot 23, desktop CopyDescriptors(destCount, destStarts, destSizes, srcCount, srcStarts, srcSizes, type).
void __stdcall device_copy_descriptors(Wrapper* self, UINT dest_count, const D3D12_CPU_DESCRIPTOR_HANDLE* dest_starts,
                                       const UINT* dest_sizes, UINT src_count, const D3D12_CPU_DESCRIPTOR_HANDLE* src_starts,
                                       const UINT* src_sizes, D3D12_DESCRIPTOR_HEAP_TYPE type)
{
    // Validate each complete logical range before the first desktop copy.
    const auto valid = [](UINT count, const D3D12_CPU_DESCRIPTOR_HANDLE* starts, const UINT* sizes, u64& total) {
        for (UINT i = 0; i < count; ++i) {
            const UINT size = sizes ? sizes[i] : 1;
            UINT available;
            desktop_cpu_handle(starts[i].ptr, nullptr, &available);
            if (size > available) return false;
            total += size;
        }
        return true;
    };
    u64 dest_total = 0, src_total = 0;
    if (!valid(dest_count, dest_starts, dest_sizes, dest_total) ||
        !valid(src_count, src_starts, src_sizes, src_total) || dest_total != src_total) return;
    const UINT increment = device_of(self)->GetDescriptorHandleIncrementSize(type);
    UINT di = 0, doff = 0, si = 0, soff = 0;
    while (di < dest_count && si < src_count) {
        const UINT dsize = dest_sizes ? dest_sizes[di] : 1, ssize = src_sizes ? src_sizes[si] : 1;
        const UINT run = (std::min)(dsize - doff, ssize - soff);
        if (run) copy_descriptor_run(self, run,
            dest_starts[di].ptr + u64{doff} * title_descriptor_stride(dest_starts[di].ptr, increment),
            src_starts[si].ptr + u64{soff} * title_descriptor_stride(src_starts[si].ptr, increment), type);
        doff += run; soff += run;
        if (doff == dsize) { ++di; doff = 0; }
        if (soff == ssize) { ++si; soff = 0; }
    }
    check_device("Device CopyDescriptors", 23);
    static LONG calls;
    if (first_calls(calls)) log("x2w-d3d12 Device::CopyDescriptors dest=%u src=%u type=%d", dest_count, src_count, type);
}
std::uint64_t __stdcall descriptor_heap_gpu_start(void* a, void* b)
{
    return small_result<D3D12_GPU_DESCRIPTOR_HANDLE>(a, b, "DescriptorHeap::GetGPUDescriptorHandleForHeapStart(no receiver)",
        [](Wrapper* w) { return static_cast<ID3D12DescriptorHeap*>(w->own().real)->GetGPUDescriptorHandleForHeapStart(); });
}
std::uint64_t __stdcall device_adapter_luid(void* a, void* b)
{
    return small_result<LUID>(a, b, "Device::GetAdapterLuid(no receiver)",
                              [](Wrapper* w) { return device_of(w)->GetAdapterLuid(); });
}

// Device slot 73 (Xbox order): (desc*, allocator, pso, riid, ppv) with a 16-byte descriptor
// (observed type 0/2 at +0, then 2, 1, 0) and an Xbox-only list IID. Creates a closed desktop
// list of the descriptor's type on the allocator so object methods have a real target; its Xbox methods are traps,
// or no-ops with X2W_D3D12_LIST_NOOP=1.
HRESULT __stdcall device_create_command_list_x(Wrapper* self, const void* desc, void* allocator, void* pso,
                                               const GUID* iid, void** out)
{
    char text[48];
    guid_text(iid, text, sizeof(text));
    char bytes[3 * 16 + 1];
    hex_text(desc, 16, bytes, sizeof(bytes));
    if (!out || !iid) return E_POINTER;
    *out = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    // Create on the allocator the title's allocator currently stands for: after a Reset that moved
    // it to another desktop allocator, the first one may still be in use.
    auto* desktop_allocator = reinterpret_cast<ID3D12CommandAllocator*>(unwrap(reinterpret_cast<u64>(allocator)));
    if (Wrapper* a = as_wrapper(allocator); a && a->own().kind == allocator_kind) {
        AcquireSRWLockExclusive(&allocator_lock);
        desktop_allocator = title_allocator(a).current->real;
        ReleaseSRWLockExclusive(&allocator_lock);
    }
    HRESULT hr = kind_for(*iid) == list_kind
        ? device_of(self)->CreateCommandList(0, desc ? *static_cast<const D3D12_COMMAND_LIST_TYPE*>(desc)
                                                     : D3D12_COMMAND_LIST_TYPE_DIRECT,
              desktop_allocator,
              reinterpret_cast<ID3D12PipelineState*>(unwrap(reinterpret_cast<u64>(pso))), IID_PPV_ARGS(&list))
        : E_NOINTERFACE;
    if (SUCCEEDED(hr)) {
        hr = list->Close();
        if (FAILED(hr)) list->Release();
    }
    static LONG calls;
    if (first_calls(calls, 4))
        log("x2w-d3d12 Device::slot73(CreateCommandList?) desc=[%s] allocator=%p pso=%p riid=%s -> 0x%08lx", bytes,
            allocator, pso, text, static_cast<unsigned long>(hr));
    if (FAILED(hr)) return hr;
    Wrapper* child = wrap(list, list_kind);
    child->own().list_device = self;
    *out = child;
    return S_OK;
}

// Device slots 78 and 79 (Xbox order): frame pacing setup. 78 receives (0, 33333, 3, 0), a 30 Hz
// frame length in microseconds and a period; 79 receives (0, offset, 0, 0). The title tolerates
// failure. Recorded and accepted; nothing paces frames yet.
UINT frame_interval_us, frame_period;
HRESULT __stdcall device_frame_interval(Wrapper*, void* output, UINT length_us, UINT period, UINT flags)
{
    frame_interval_us = length_us;
    frame_period = period;
    log("x2w-d3d12 Device::slot78(frame interval?) output=%p length=%u us period=%u flags=0x%x -> S_OK", output, length_us,
        period, flags);
    return S_OK;
}
UINT frame_event_offset_us[4];
HRESULT __stdcall device_frame_event(Wrapper*, UINT type, UINT offset_us, void* reserved, UINT flags)
{
    if (type < 4) frame_event_offset_us[type] = offset_us;
    log("x2w-d3d12 Device::slot79(frame event?) type=%u offset=%u us reserved=%p flags=0x%x -> S_OK", type, offset_us,
        reserved, flags);
    return S_OK;
}

// Device slot 133 (Xbox order): (address, D3D12_RESOURCE_DESC*, state, clear, riid, ppv), a resource
// placed at a title address. First seen with a row-major buffer desc (flags 0x20004) over a
// placeholder-replaced XMem region, state 0xffffffff, a null clear value and the ID3D12Resource IID.
// Desktop: the allocation containing the address is opened as an existing heap (cached) and a
// cross-adapter buffer is placed at the address's offset. Its desktop GPU address differs from the
// title address; nothing translates title GPU addresses yet. Textures trap.
std::mutex placed_mutex;
struct ExistingHeap {
    ID3D12Heap* real;
    x2w_budget::Ticket* budget;
};
std::vector<std::pair<void*, ExistingHeap>> existing_heaps;

IndirectAddressAlias indirect_address_alias(Wrapper* device, u64 address, u64 bytes)
{
    static const bool enabled = x2w_local::env_on("X2W_TRANSLATED_INDIRECT_ADDRESS_ALIAS");
    if (!enabled || !device || device != the_device || !address || !bytes || bytes > UINT64_MAX - address)
        return {};
    constexpr u64 page_bytes = 65536;
    const u64 page = address & ~(page_bytes - 1);
    const u64 offset = address - page;
    if (bytes > page_bytes - offset) return {};
    x2w_startup::PlacedBufferRange memory{};
    if (!x2w_startup::measure_placed_buffer(reinterpret_cast<void*>(page), page_bytes, memory) ||
        !memory.covered || memory.type != MEM_PRIVATE) return {};
    const auto nearby = placed_buffer_range(device, page, page_bytes, true);
    if (nearby.reject || nearby.owner) return {};

    // Reuse the existing allocation heap opened by slot 133. It is already held
    // for the startup device's lifetime. The caller keeps this address allocation
    // live through the GPU work and must not activate an overlapping placed alias.
    std::lock_guard lock(placed_mutex);
    const auto known = std::find_if(existing_heaps.begin(), existing_heaps.end(),
        [&](const auto& entry) { return entry.first == memory.base; });
    if (known == existing_heaps.end()) return {};
    auto* heap = known->second.real;
    struct Page { ID3D12Heap* heap; u64 page; x2w_startup::ComPtr<ID3D12Resource> resource; };
    static std::array<Page, 64> pages{};
    auto it = std::find_if(pages.begin(), pages.end(), [&](const Page& entry) { return entry.heap == heap && entry.page == page; });
    if (it != pages.end()) return {it->resource.Get(), offset};
    it = std::find_if(pages.begin(), pages.end(), [](const Page& entry) { return !entry.heap; });
    if (it == pages.end()) return {};
    x2w_startup::ComPtr<ID3D12Device> owner;
    auto* desktop = device_of(device);
    if (FAILED(heap->GetDevice(IID_PPV_ARGS(&owner))) || owner.Get() != desktop) return {};
    const auto base = reinterpret_cast<u64>(memory.base);
    if (base > page || (page - base) % page_bytes) return {};
    const u64 heap_size = heap->GetDesc().SizeInBytes;
    if (page - base > heap_size || page_bytes > heap_size - (page - base)) return {};
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, page_bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN,
        {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER};
    if (FAILED(desktop->CreatePlacedResource(heap, page - base, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
        IID_PPV_ARGS(&it->resource)))) return {};
    it->heap = heap;
    it->page = page;
    return {it->resource.Get(), offset};
}
volatile LONG64 placed_buffer_calls, placed_buffer_region_info, placed_buffer_queries;  // five-second totals
// LOCAL: the address ranges of the private allocations opened as desktop heaps. A memory query
// costs over a millisecond in this process (very many mapped sections); desktop_va and
// desktop_range ask this table first and query the kernel only for addresses outside it.
struct PrivateRange { u64 base, end; };
SRWLOCK private_range_lock = SRWLOCK_INIT;
std::vector<PrivateRange> private_ranges;
void note_private_range(void* base, ID3D12Heap* heap)
{
    if (!base || !heap) return;
    const u64 size = heap->GetDesc().SizeInBytes;
    AcquireSRWLockExclusive(&private_range_lock);
    private_ranges.push_back({reinterpret_cast<u64>(base), reinterpret_cast<u64>(base) + size});
    ReleaseSRWLockExclusive(&private_range_lock);
}
// The allocation base and end of a private allocation opened as a heap that contains address.
bool known_private(u64 address, u64* base, u64* end)
{
    bool found = false;
    AcquireSRWLockShared(&private_range_lock);
    for (const PrivateRange& r : private_ranges)
        if (address >= r.base && address < r.end) {
            *base = r.base;
            *end = r.end;
            found = true;
            break;
        }
    ReleaseSRWLockShared(&private_range_lock);
    return found;
}
ID3D12Heap* existing_heap(void* base)
{
    std::lock_guard lock(placed_mutex);
    for (auto& [known, h] : existing_heaps)
        if (known == base) return h.real;
    if (!x2w_startup::private_heap_importable(base)) return nullptr;
    ID3D12Device3* device3 = nullptr;
    ID3D12Heap* heap = nullptr;
    HRESULT hr = device_of(the_device)->QueryInterface(IID_PPV_ARGS(&device3));
    if (SUCCEEDED(hr)) {
        hr = device3->OpenExistingHeapFromAddress(base, IID_PPV_ARGS(&heap));
        device3->Release();
    }
    if (FAILED(hr)) {
        log("x2w-va OpenExistingHeapFromAddress(%p) -> 0x%08lx", base, static_cast<unsigned long>(hr));
        return nullptr;
    }
    existing_heaps.emplace_back(base, ExistingHeap{heap, nullptr});  // LOCAL desktop_va path: no budget ticket
    note_private_range(base, heap);
    return heap;
}
// LOCAL: title GPU address -> desktop GPU address. Title GPU memory is the title's own (CPU) memory.
// A private (VirtualAlloc) allocation can be opened as a desktop heap; buffers placed on that heap
// in 256 MiB steps, each 512 MiB long (or up to the heap's end), let any range below 256 MiB be
// addressed by one buffer. The desktop GPU then reads and writes the same pages as the title's CPU.
struct VaWindow {
    void* base;
    UINT64 start;
    ID3D12Resource* buffer;
    UINT64 width;
};
SRWLOCK va_lock = SRWLOCK_INIT;
std::vector<VaWindow>& va_windows()
{
    static std::vector<VaWindow> windows;
    return windows;
}
LONG va_mapped, va_failed;
ID3D12Heap* existing_heap(void* base);  // opens (once) the desktop heap over a private allocation
// A chunk section opened once as a desktop heap (nullptr when it cannot be); shared by section_va and, with
// X2W_MEMCUT_ALIAS_SLABS, by placed buffers over pool slabs (device slot 133).
ID3D12Heap* section_heap(HANDLE section)
{
    static SRWLOCK lock = SRWLOCK_INIT;
    static std::vector<std::pair<HANDLE, ID3D12Heap*>> heaps;
    AcquireSRWLockExclusive(&lock);
    for (const auto& [h, hp] : heaps)
        if (h == section) {
            ReleaseSRWLockExclusive(&lock);
            return hp;
        }
    ID3D12Heap* heap = nullptr;
    ID3D12Device3* device3 = nullptr;
    HRESULT hr = device_of(the_device)->QueryInterface(IID_PPV_ARGS(&device3));
    if (SUCCEEDED(hr)) {
        hr = device3->OpenExistingHeapFromFileMapping(section, IID_PPV_ARGS(&heap));
        device3->Release();
    }
    log("x2w-va heap over chunk section %p -> 0x%08lx size 0x%llx", section, static_cast<unsigned long>(hr),
        SUCCEEDED(hr) ? heap->GetDesc().SizeInBytes : 0ull);
    if (FAILED(hr)) heap = nullptr;
    heaps.emplace_back(section, heap);
    ReleaseSRWLockExclusive(&lock);
    return heap;
}
// A chunk section opened once as a desktop heap; buffers placed on it in 256 MiB steps (512 MiB long).
D3D12_GPU_VIRTUAL_ADDRESS section_va(HANDLE section, ULONG64 offset)
{
    static SRWLOCK lock = SRWLOCK_INIT;
    static std::vector<std::tuple<HANDLE, ULONG64, ID3D12Resource*>> windows;
    const ULONG64 start = offset & ~((1ull << 28) - 1);
    AcquireSRWLockExclusive(&lock);
    for (const auto& [h, s, b] : windows)
        if (h == section && s == start) {
            const D3D12_GPU_VIRTUAL_ADDRESS va = b ? b->GetGPUVirtualAddress() + (offset - start) : 0;
            ReleaseSRWLockExclusive(&lock);
            return va;
        }
    ID3D12Heap* heap = section_heap(section);
    HRESULT hr = S_OK;
    ID3D12Resource* buffer = nullptr;
    if (heap) {
        const UINT64 size = heap->GetDesc().SizeInBytes;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = size - start < (2ull << 28) ? size - start : (2ull << 28);
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
        hr = device_of(the_device)->CreatePlacedResource(heap, start, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&buffer));
        if (FAILED(hr)) buffer = nullptr;
        log("x2w-va chunk window %p+0x%llx -> 0x%08lx va 0x%llx", section, start, static_cast<unsigned long>(hr),
            buffer ? buffer->GetGPUVirtualAddress() : 0ull);
    }
    windows.emplace_back(section, start, buffer);
    ReleaseSRWLockExclusive(&lock);
    note_window(buffer);
    return buffer ? buffer->GetGPUVirtualAddress() + (offset - start) : 0;
}
D3D12_GPU_VIRTUAL_ADDRESS desktop_va(u64 address)
{
    if (!address) return 0;
    MEMORY_BASIC_INFORMATION info{};
    u64 known_base = 0, known_end = 0;
    const bool known = known_private(address, &known_base, &known_end);
    if (known) {
        info.AllocationBase = reinterpret_cast<void*>(known_base);
        info.Type = MEM_PRIVATE;
        info.State = MEM_COMMIT;
    } else if (timed_query(reinterpret_cast<const void*>(address), &info) != sizeof(info)) {
        info = {};
    }
    if (info.Type == MEM_MAPPED) {
        // Physical pages in 1 GiB chunk sections (xmem X2W_XMEM_CHUNKS=1): a heap over the chunk.
        using Location = BOOL(WINAPI*)(const void*, HANDLE*, ULONG64*);
        static const Location location = [] {
            const HMODULE xmem = GetModuleHandleW(L"xmem.dll");
            return xmem ? reinterpret_cast<Location>(GetProcAddress(xmem, "x2w_xmem_gpu_location")) : nullptr;
        }();
        HANDLE section = nullptr;
        ULONG64 at = 0;
        if (location && location(reinterpret_cast<const void*>(address), &section, &at))
            return section_va(section, at);
        if (x2w_local::first_of(InterlockedIncrement(&va_failed), 8)) log("x2w-va 0x%llx: mapped page without a chunk location", address);
        return 0;
    }
    if (info.Type != MEM_PRIVATE || info.State != MEM_COMMIT) {
        if (x2w_local::first_of(InterlockedIncrement(&va_failed), 8))
            log("x2w-va 0x%llx: not a committed private allocation (type 0x%lx state 0x%lx)", address, info.Type, info.State);
        return 0;
    }
    void* base = info.AllocationBase;
    const UINT64 offset = address - reinterpret_cast<u64>(base);
    const UINT64 start = offset & ~((1ull << 28) - 1);
    const auto lookup = [&]() -> D3D12_GPU_VIRTUAL_ADDRESS {
        for (const auto& w : va_windows())
            if (w.base == base && w.start == start && offset - start < w.width)
                return w.buffer->GetGPUVirtualAddress() + (offset - start);
        return 0;
    };
    AcquireSRWLockShared(&va_lock);
    const auto cached = lookup();
    ReleaseSRWLockShared(&va_lock);
    if (cached) return cached;
    ID3D12Heap* heap = existing_heap(base);
    if (!heap) return 0;
    const auto size = heap->GetDesc().SizeInBytes;
    if (offset >= size) return 0;
    AcquireSRWLockExclusive(&va_lock);
    const auto concurrent = lookup();
    if (concurrent) {
        ReleaseSRWLockExclusive(&va_lock);
        return concurrent;
    }
    ID3D12Resource* buffer = nullptr;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = (std::min<UINT64>)(size - start, 2ull << 28);
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
    const HRESULT hr = device_of(the_device)->CreatePlacedResource(heap, start, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&buffer));
    if (SUCCEEDED(hr)) {
        note_window(buffer);
        va_windows().push_back({base, start, buffer, desc.Width});
    }
    ReleaseSRWLockExclusive(&va_lock);
    const LONG n = SUCCEEDED(hr) ? InterlockedIncrement(&va_mapped) : InterlockedIncrement(&va_failed);
    if (x2w_local::first_of(n, 16))
        log("x2w-va window base=%p start=0x%llx heap=%p -> 0x%08lx desktop va 0x%llx", base, start, static_cast<void*>(heap),
            static_cast<unsigned long>(hr), SUCCEEDED(hr) ? buffer->GetGPUVirtualAddress() : 0ull);
    return SUCCEEDED(hr) ? buffer->GetGPUVirtualAddress() + (offset - start) : 0;
}


HRESULT __stdcall device_place_resource_x(Wrapper* self, void* address, const D3D12_RESOURCE_DESC* desc, UINT state,
                                          const void* clear, const GUID* iid, void** out)
{
    // LOCAL timers: the whole call ("placed") and the gap since this thread's previous call ("placed-gap").
    static thread_local LONG64 placed_last_end = 0;
    struct PlacedEnd {
        LARGE_INTEGER start;
        LONG64* last_end;
        ~PlacedEnd()
        {
            LARGE_INTEGER end;
            QueryPerformanceCounter(&end);
            InterlockedAdd64(&time_ticks[time_placed], end.QuadPart - start.QuadPart);
            InterlockedIncrement64(&time_calls[time_placed]);
            InterlockedAdd64(&placed_ticks_total, end.QuadPart - start.QuadPart);
            InterlockedIncrement64(&placed_calls_total);
            *last_end = end.QuadPart;
        }
    } placed_end{{}, &placed_last_end};
    QueryPerformanceCounter(&placed_end.start);
    {
        static const LONG64 frequency = [] {
            LARGE_INTEGER f;
            QueryPerformanceFrequency(&f);
            return f.QuadPart;
        }();
        const LONG64 gap = placed_end.start.QuadPart - placed_last_end;
        if (placed_last_end && gap >= 0 && gap < frequency) {
            InterlockedAdd64(&time_ticks[time_placed_gap], gap);
            InterlockedIncrement64(&time_calls[time_placed_gap]);
        }
    }
    char text[48];
    guid_text(iid, text, sizeof(text));
    if (!out || !desc || !iid) return E_POINTER;
    *out = nullptr;
    check_device("Device before slot", 133);
    if (kind_for(*iid) != resource_kind) return E_NOINTERFACE;
    if (desc->Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) {
        // Textures cannot be placed on title memory on desktop: a committed default-heap texture
        // stands in, not aliased with the title address. The clear value is dropped if refused.
        D3D12_RESOURCE_DESC texture = sanitized(*desc);
        if (x2w_startup::supports_bc7_4k_s(*desc) && x2w_local::env_on("X2W_XG_BC7_4K_S_LAYOUT")) {
            const x2w_startup::Bc16Surface surface(*desc);
            // Desktop BC7 allocations require whole blocks at the top mip.
            texture.Width = surface.columns * x2w_startup::Bc16Surface::block_texels;
            texture.Height = surface.rows * x2w_startup::Bc16Surface::block_texels;
        }
        const D3D12_HEAP_PROPERTIES props{D3D12_HEAP_TYPE_DEFAULT};
        // LOCAL merge: the clear format is mapped to its desktop format first (local), then upstream's packed stand-in.
        D3D12_CLEAR_VALUE mapped_clear{};
        if (clear) { mapped_clear = *static_cast<const D3D12_CLEAR_VALUE*>(clear); mapped_clear.Format = desktop_format(mapped_clear.Format); }
        D3D12_CLEAR_VALUE clear_copy{};
        const auto* clear_value = packed_clear_value(clear ? &mapped_clear : nullptr,
                                                    desktop_format(desc->Format), texture.Format, clear_copy);
        ID3D12Resource* created = nullptr;
        HRESULT texture_hr = device_of(self)->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &texture,
                                                                      D3D12_RESOURCE_STATE_COMMON, clear_value,
                                                                      IID_PPV_ARGS(&created));
        if (FAILED(texture_hr) && clear_value)
            texture_hr = device_of(self)->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &texture,
                                                                  D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&created));
        static LONG textures, failures;
        if (first_calls(textures, 6) || (FAILED(texture_hr) && first_calls(failures, 8))) {
            char bytes[3 * 0x38 + 1];
            hex_text(desc, 0x38, bytes, sizeof(bytes));
            log("x2w-d3d12 Device::slot133(CreatePlacedResourceX?) texture address=%p dim=%d %llux%ux%u mips=%u fmt=%d "
                "samples=%u flags=0x%x layout=0x%x state=0x%x clear=%p -> 0x%08lx not aliased [%s]", address, desc->Dimension,
                desc->Width, desc->Height, desc->DepthOrArraySize, desc->MipLevels, desc->Format, desc->SampleDesc.Count,
                desc->Flags, desc->Layout, state, clear, static_cast<unsigned long>(texture_hr),
                FAILED(texture_hr) ? bytes : "");
        }
        if (FAILED(texture_hr)) return texture_hr;
        x2w_startup::mark_packed_stand_in(created, desktop_format(desc->Format));
        Wrapper* w = wrap(created, resource_kind);
        if (texture.Format != desc->Format && texture.Format == x2w_srgb::family(desc->Format))
            w->own().srgb_uav_format = static_cast<unsigned char>(desc->Format);
        w->own().budget = x2w_budget::committed(device_of(self), created, props.Type);
        w->own().title_memory = address;
        if (cold_textures_enabled() && address && texture.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
            texture.SampleDesc.Count == 1 && texture.Flags == D3D12_RESOURCE_FLAG_NONE)
            InterlockedExchange(&w->own().cold_texture_state, priority_normal);
        note_resource(w, desc, 0);
        retire_set_class(w, props, D3D12_HEAP_FLAG_NONE, texture, address, /*allow_title_memory=*/true);
        *out = w;
        return S_OK;
    }
    // The allocation holding the address, and whether every byte from its base to the end of the
    // buffer is committed in it: one region-information call for a fully committed private
    // allocation, otherwise a region walk that stops at the end of the buffer. The GPU cannot fault in a lazily
    // mapped physical page (X2W_XMEM_LAZY_COMMIT), so the buffer's pages are committed first; a lazy startup
    // block (X2W_XMEM_LAZY_STARTUP) is committed whole, as the heap opened on it is reused for the block.
    x2w_local::commit_lazy(address, static_cast<std::size_t>(desc->Width));
    x2w_local::commit_lazy_block(address);
    InterlockedIncrement64(&placed_buffer_calls);  // LOCAL walk below (no measure_placed_buffer)
    TimedPart query_time(time_placed_query);
    MEMORY_BASIC_INFORMATION info{};
    if (!address || VirtualQuery(address, &info, sizeof(info)) != sizeof(info) || info.State != MEM_COMMIT) {
        query_time.stop();
        log("x2w-d3d12 Device::slot133(CreatePlacedResourceX?) address=%p not committed -> E_INVALIDARG", address);
        return E_INVALIDARG;
    }
    void* base = info.AllocationBase;  // LOCAL walk (upstream: measure_placed_buffer range.base)
    const auto initial_state = state == D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE ||
        state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS ? static_cast<D3D12_RESOURCE_STATES>(state) : D3D12_RESOURCE_STATE_COMMON;
    if (initial_state != D3D12_RESOURCE_STATE_COMMON && !(desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
        return E_INVALIDARG;
    const UINT64 offset = static_cast<char*>(address) - static_cast<char*>(base);
    ID3D12Heap* heap = nullptr;
    x2w_budget::Ticket* backing_budget = nullptr;
    HRESULT hr = S_OK;
    // Desktop only accepts 64 KiB-aligned offsets, a private (VirtualAlloc) allocation, and a buffer
    // that lies inside that one allocation. Attempts that cannot succeed are skipped: each failure
    // throws inside D3D12Core.
    // LOCAL (perf_work/STARTUP.md): the allocation walk below queries every region from the allocation
    // base; one query on a large touched region costs about 1.8 ms per GiB (scratch bench), and the
    // start-up loop places ~2,800 small buffers at unaligned offsets in a 2.2 GiB block, where the walk
    // result is never used. The checks that need no walk come first; the result is the same.
    // X2W_D3D12_PLACED_WALK=1 walks for every call, as before.
    static const bool always_walk = x2w_local::env_on("X2W_D3D12_PLACED_WALK");
    const bool rejected = (offset & 0xFFFF) || info.Type != MEM_PRIVATE ||
                          initial_state == D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
    MEMORY_BASIC_INFORMATION region{};
    SIZE_T allocation_bytes = 0;
    static const bool placed_fast = x2w_local::env_on("X2W_PLACED_FAST");
    bool fast = false;
    if (placed_fast && !rejected && !always_walk) {
        const auto* region_end = static_cast<const char*>(info.BaseAddress) + info.RegionSize;
        if (static_cast<const char*>(address) + desc->Width <= region_end) {
            UINT64 heap_bytes = 0;
            {
                std::lock_guard lock(placed_mutex);
                for (auto& [known, h] : existing_heaps)
                    if (known == base) heap_bytes = h.real->GetDesc().SizeInBytes;
            }
            if (heap_bytes && offset <= heap_bytes && desc->Width <= heap_bytes - offset) {
                fast = true;
                allocation_bytes = static_cast<SIZE_T>(heap_bytes);
                static LONG fast_calls;
                const LONG n = InterlockedIncrement(&fast_calls);
                if (x2w_local::first_of(n, 3) || n % 10000 == 0)
                    log("x2w-d3d12 slot133 X2W_PLACED_FAST: %ld calls placed without the allocation walk (heap %llu bytes)", n, heap_bytes);
            }
        }
    }
    if ((!rejected || always_walk) && !fast)
        for (auto* cursor = static_cast<char*>(base);
             VirtualQuery(cursor, &region, sizeof(region)) && region.AllocationBase == base && region.State == MEM_COMMIT;
             cursor += region.RegionSize)
            allocation_bytes += region.RegionSize;
    // LOCAL memcut: X2W_MEMCUT_ALIAS_SLABS=1 places a buffer over pool-slab pages on the slab's section heap.
    UINT64 slab_offset = 0;
    ID3D12Heap* const slab_heap = info.Type == MEM_MAPPED && initial_state != D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE
                                      ? memcut_slab_heap(address, desc->Width, &slab_offset) : nullptr;
    if (!slab_heap && (rejected || offset > allocation_bytes || desc->Width > allocation_bytes - offset)) hr = E_INVALIDARG;
    query_time.stop();
    TimedPart create_time(time_placed_create);
    if (slab_heap) heap = slab_heap;
    else if (SUCCEEDED(hr)) {
        std::lock_guard lock(placed_mutex);
        for (const auto& [known, h] : existing_heaps) {
            if (known == base) {
                heap = h.real;
                backing_budget = h.budget;
            }
        }
        if (!heap && !x2w_startup::private_heap_importable(base)) hr = E_INVALIDARG;
        if (!heap && SUCCEEDED(hr)) {
            ID3D12Device3* device3 = nullptr;
            hr = device_of(self)->QueryInterface(IID_PPV_ARGS(&device3));
            if (SUCCEEDED(hr)) {
                hr = device3->OpenExistingHeapFromAddress(base, IID_PPV_ARGS(&heap));
                device3->Release();
            }
            if (SUCCEEDED(hr)) {
                backing_budget = x2w_budget::backing_heap(heap, 0);
                existing_heaps.emplace_back(base, ExistingHeap{heap, backing_budget});
                note_private_range(base, heap);
            }
        }
    }
    D3D12_RESOURCE_DESC copy = *desc;
    copy.Flags = static_cast<D3D12_RESOURCE_FLAGS>((sanitized(*desc).Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) |
                                                   D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER);
    ID3D12Resource* resource = nullptr;
    if (SUCCEEDED(hr))
        hr = device_of(self)->CreatePlacedResource(heap, slab_heap ? slab_offset : offset, &copy, initial_state, nullptr,
                                                   IID_PPV_ARGS(&resource));
    // A buffer that spans several allocations (adjacent placeholder pieces) cannot be placed on one
    // existing heap. Fallback: a committed default-heap buffer, not aliased with the title memory.
    const bool aliased = SUCCEEDED(hr);
    const char* placement = slab_heap && aliased ? "aliased slab" : "aliased";  // LOCAL memcut: "aliased slab"
    if (slab_heap && !aliased) memcut_slab_failed(address, desc->Width, hr);
    if (FAILED(hr)) {
        const D3D12_HEAP_PROPERTIES props{D3D12_HEAP_TYPE_DEFAULT};
        copy.Flags = static_cast<D3D12_RESOURCE_FLAGS>(desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        static x2w_startup::PlacedPool pool;  // LOCAL: X2W_D3D12_PLACED_POOL=1 (placed_pool.h)
        hr = pool.create(device_of(self), copy, initial_state, &resource);
        placement = "not aliased pooled";
        if (FAILED(hr)) {
            hr = device_of(self)->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &copy, initial_state,
                                                          nullptr, IID_PPV_ARGS(&resource));
            placement = "not aliased";
        }
    }
    create_time.stop();
    TimedPart log_time(time_placed_log);
    log("x2w-d3d12 Device::slot133(CreatePlacedResourceX?) address=%p base=%p offset=0x%llx width=0x%llx flags=0x%x "
        "state=0x%x clear=%p riid=%s -> 0x%08lx %s gpuva=0x%llx", address, base, offset, desc->Width, desc->Flags, state,
        clear, text, static_cast<unsigned long>(hr), placement, resource ? resource->GetGPUVirtualAddress() : 0ull);
    if (const int r = range_watch_hit(reinterpret_cast<u64>(address), desc->Width); r >= 0 && range_watch_line(r))
        log("x2w-watch 0x%llx: slot133 placed buffer at %p width 0x%llx flags 0x%x -> 0x%08lx %s frame %ld", range_watch_address(r), address,
            desc->Width, desc->Flags, static_cast<unsigned long>(hr), placement, frame_number);
    if (range_watch_count()) range_watch_sample("slot133-create");
    log_time.stop();
    if (FAILED(hr)) return hr;
    Wrapper* w = wrap(resource, resource_kind);
    if (release_retired_enabled()) w->own().retire_never = static_cast<std::uint8_t>(RetireNever::buffer);
    w->own().title_memory = address;
    const ResourceInfo record{*desc, desc->Width, aliased ? Placement::aliased_buffer : Placement::copied_buffer, self, initial_state};
    auto& group = placement_group(desc->Width);
    hr = x2w_startup::publish_placement(note_lock, resource_infos, group.by_start, group.max_width,
        static_cast<const Wrapper*>(w), reinterpret_cast<u64>(address), record, [&] { index_resource(w, record); });
    if (FAILED(hr)) {
        // The caller has never received this wrapper; release its desktop resource and storage.
        AcquireSRWLockExclusive(&registry_lock);
        registry().erase(w);
        ReleaseSRWLockExclusive(&registry_lock);
        resource->Release();
        _aligned_free(&w->own());
        return hr;
    }
    if (aliased) {
        w->own().budget = x2w_budget::resource(device_of(self), resource, x2w_budget::placed);
        w->own().backing_budget = backing_budget;
        if (backing_budget) backing_budget->begin_use();
    } else {
        w->own().budget = x2w_budget::committed(device_of(self), resource, D3D12_HEAP_TYPE_DEFAULT);
    }
    *out = w;
    return S_OK;
}

#include "d3d12_memory_srv.inl"
#include "startup_stubs/list_copy.inl"

// X2W_D3D12_RELEASE_RETIRED=1 (docs/startup-runtime.md#d3d12-resource-budget): the desktop object of a retire class
// member is released after its final guest Release once (1) the release-time fence of every queue has passed (each queue
// owns a private fence that the Release signals) and (2) no list or submission names it: a list holds a desktop
// reference from a forwarded copy until it is Reset, fails to Close or is destroyed, and each submission holds one until
// its queue passes a fence signalled behind it. Nothing waits; the drain only try-locks and drops references outside
// the lock. What cannot be recorded stays retained for the process, the lifetime without the switch.
constexpr LONG retired_refs = -0x40000000;
SRWLOCK retire_lock = SRWLOCK_INIT;  // a leaf: taken after draw_lock and allocator_lock, nothing is taken under it
struct RetireQueue {
    ID3D12CommandQueue* queue;  // no reference: release() never frees a queue object
    ID3D12Fence* fence;  // null: no marker for this queue, so nothing retires
    UINT64 signaled;
};
struct RetiredResource {
    Wrapper* wrapper;
    std::vector<UINT64> targets;  // per RetireQueue that existed at the Release
};
struct RetireFlight {
    std::size_t queue;
    UINT64 value;
    std::vector<ID3D12Resource*> resources;  // one desktop reference each
};
std::vector<RetireQueue> retire_queues;
std::vector<RetiredResource> retired_resources;
std::vector<RetireFlight> retire_flights;
std::unordered_map<Wrapper*, std::vector<ID3D12Resource*>> retire_list_holds;  // one desktop reference per entry
volatile LONG retire_pending;  // retired_resources and retire_flights, read without the lock
volatile LONG retire_holds;    // entries of retire_list_holds, read without the lock
volatile LONG64 retire_freed, retire_kept;

void register_retire_queue(Wrapper* queue)
{
    if (!release_retired_enabled() || !queue || !queue->own().real) return;
    auto* real = static_cast<ID3D12CommandQueue*>(queue->own().real);
    ID3D12Fence* fence = nullptr;
    ID3D12Device* device = nullptr;
    if (SUCCEEDED(real->GetDevice(IID_PPV_ARGS(&device)))) {
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) fence = nullptr;
        device->Release();
    }
    AcquireSRWLockExclusive(&retire_lock);
    try {
        retire_queues.push_back({real, fence, 0});
    } catch (const std::bad_alloc&) {
        if (fence) fence->Release();
    }
    ReleaseSRWLockExclusive(&retire_lock);
}

// A use the holds do not cover (a view, residency, presentation) takes the resource out of the class for good.
void retire_leave_class(const void* object, RetireLeft why)
{
    if (!release_retired_enabled()) return;
    if (Wrapper* w = as_wrapper(object)) {
        if (w->own().retire_when_idle) w->own().retire_left = static_cast<std::uint8_t>(why);  // the first use counts
        w->own().retire_when_idle = false;
    }
}

void retire_add_view_ref(const void* object)
{
    if (!release_retired_enabled()) return;
    if (Wrapper* w = as_wrapper(object); w && w->own().kind == resource_kind)
        InterlockedIncrement(&w->own().retire_view_refs);
}

void retire_release_view_ref(const void* object)
{
    if (!release_retired_enabled()) return;
    if (Wrapper* w = as_wrapper(object); w && w->own().kind == resource_kind)
        InterlockedDecrement(&w->own().retire_view_refs);
}

void retire_drop_heap_views(Wrapper* heap)
{
    if (!release_retired_enabled() || !heap) return;
    AcquireSRWLockExclusive(&note_lock);
    for (HeapInfo& h : heap_infos()) {
        if (h.wrapper != reinterpret_cast<u64>(heap) || !h.shadow) continue;
        for (std::uint32_t i = 0; i < h.count; ++i) {
            const std::uint32_t id = h.shadow[i];
            if (!id) continue;
            h.shadow[i] = 0;
            if (id - 1 < view_records().size() && view_records()[id - 1].slot == 18 && view_materialized()[id - 1])
                retire_release_view_ref(reinterpret_cast<const void*>(view_records()[id - 1].resource));
        }
    }
    ReleaseSRWLockExclusive(&note_lock);
}

bool retire_address_has_live_user(u64 address, const Wrapper* skip)
{
    bool other_live = false;
    AcquireSRWLockShared(&note_lock);
    resource_address_index().for_each_reaching(address, 1, [&](const Wrapper* wrapper) {
        if (wrapper == skip || reinterpret_cast<u64>(wrapper->own().title_memory) != address) return false;
        if (InterlockedCompareExchange(const_cast<LONG*>(&wrapper->own().refs), 0, 0) > 0 ||
            InterlockedCompareExchange(const_cast<LONG*>(&wrapper->own().retire_view_refs), 0, 0) > 0) {
            other_live = true;
            return true;
        }
        return false;
    });
    ReleaseSRWLockShared(&note_lock);
    if (other_live) return true;
    std::lock_guard lock(tiled_srv_mutex);
    return tiled_srv_addresses.count(address) != 0;
}

void retire_note_title_release(Wrapper* w)
{
    const u64 address = reinterpret_cast<u64>(w->own().title_memory);
    if (!address) return;
    const bool shared_addr = retire_address_has_live_user(address, w);
    const bool no_views = InterlockedCompareExchange(&w->own().retire_view_refs, 0, 0) == 0;
    AcquireSRWLockExclusive(&written_lock);
    if (const auto found = written_textures().find(address); found != written_textures().end()) {
        if (!shared_addr && no_views) {
            written_textures().erase(found);
        } else {
            w->own().retire_has_written_gen = true;
            w->own().retire_written_gen = found->second.generation;
        }
    }
    ReleaseSRWLockExclusive(&written_lock);
}

// After a copy, texture copy or resolve was forwarded to the desktop list: the list keeps a desktop reference to the
// resource until its recording is dropped. Only wrappers are members of the class, so a raw desktop pointer is never
// held. Assumption, not proven: the guest has no raw pointer to the desktop object of a class member.
void retire_hold(Wrapper* list, Wrapper* resource)
{
    warm_texture(resource);  // Copies and resolves can bind wrappers directly.
    if (!resource || !resource->own().retire_when_idle) return;
    bool held = false;
    AcquireSRWLockExclusive(&retire_lock);
    // own().real is cleared by drain_retired_resources under this lock, so the read and the AddRef happen under it
    // (retire_lock is a leaf: nothing else is taken inside).
    auto* real = static_cast<ID3D12Resource*>(resource->own().real);
    if (real) {
        try {
            auto& held_by_list = retire_list_holds[list];
            if (std::find(held_by_list.begin(), held_by_list.end(), real) != held_by_list.end()) {
                held = true;
            } else {
                held_by_list.reserve(held_by_list.size() + 1);
                real->AddRef();
                held_by_list.push_back(real);
                InterlockedIncrement(&retire_holds);
                held = true;
            }
        } catch (const std::bad_alloc&) {
        }
    }
    if (!held) {
        // The use cannot be recorded: leave the class, and mark it so the drain keeps the desktop object for the
        // process (as it keeps a submission it could not fence) rather than dropping one a list still names.
        resource->own().retire_when_idle = false;
        resource->own().retire_left = static_cast<std::uint8_t>(RetireLeft::hold_missed);
        resource->own().retire_hold_missed = true;
    }
    ReleaseSRWLockExclusive(&retire_lock);
}

void retire_discard_holds(Wrapper* list)
{
    if (!InterlockedCompareExchange(&retire_holds, 0, 0)) return;
    std::vector<ID3D12Resource*> dropped;
    AcquireSRWLockExclusive(&retire_lock);
    if (const auto found = retire_list_holds.find(list); found != retire_list_holds.end()) {
        dropped = std::move(found->second);
        retire_list_holds.erase(found);
        InterlockedAdd(&retire_holds, -static_cast<LONG>(dropped.size()));
    }
    ReleaseSRWLockExclusive(&retire_lock);
    for (ID3D12Resource* resource : dropped) resource->Release();
}

// Right after ExecuteCommandLists: the lists' held resources stay referenced until the queue passes a fence signalled
// behind the submission. If that fence cannot be recorded or signalled the references are kept for the process.
void retire_note_submission(ID3D12CommandQueue* queue, const std::vector<Wrapper*>& lists)
{
    if (!InterlockedCompareExchange(&retire_holds, 0, 0)) return;
    std::vector<ID3D12Resource*> named;
    AcquireSRWLockExclusive(&retire_lock);
    bool room = true;
    try {
        for (Wrapper* list : lists) {
            const auto found = retire_list_holds.find(list);
            if (found == retire_list_holds.end()) continue;
            for (ID3D12Resource* resource : found->second)
                if (std::find(named.begin(), named.end(), resource) == named.end()) named.push_back(resource);
        }
        retire_flights.reserve(retire_flights.size() + 1);
    } catch (const std::bad_alloc&) {
        room = false;
    }
    bool recorded = false;
    if (room && !named.empty()) {
        std::size_t index = 0;
        while (index < retire_queues.size() && retire_queues[index].queue != queue) ++index;
        if (index < retire_queues.size() && retire_queues[index].fence) {
            RetireQueue& q = retire_queues[index];
            if (SUCCEEDED(queue->Signal(q.fence, q.signaled + 1))) {
                for (ID3D12Resource* resource : named) resource->AddRef();
                ++q.signaled;
                retire_flights.push_back({index, q.signaled, std::move(named)});
                InterlockedIncrement(&retire_pending);
                recorded = true;
            }
        }
    }
    if (!recorded && (!room || !named.empty())) {
        // Keep everything the lists name for the process; no allocation is needed to do so.
        for (Wrapper* list : lists)
            if (const auto found = retire_list_holds.find(list); found != retire_list_holds.end())
                for (ID3D12Resource* resource : found->second) resource->AddRef();
        InterlockedIncrement64(&retire_kept);
    }
    ReleaseSRWLockExclusive(&retire_lock);
    if (!recorded && (!room || !named.empty())) {
        static LONG reported;
        if (first_calls(reported, 4)) log("x2w-d3d12 retire: a submission could not be fenced; its resources stay retained");
    }
}

// Called by the final Release of a resource of the class.
void retire_resource(Wrapper* w)
{
    if (w->own().kind != resource_kind || !w->own().real) return;
    RetiredResource item{w, {}};
    bool queued = false;
    AcquireSRWLockExclusive(&retire_lock);
    try {
        // Without a registered queue nothing orders a marker: keep the resource. Every queue needs its fence.
        queued = !retire_queues.empty();
        for (const RetireQueue& q : retire_queues) queued = queued && q.fence;
        if (queued) {
            item.targets.reserve(retire_queues.size());
            retired_resources.reserve(retired_resources.size() + 1);
            for (RetireQueue& q : retire_queues) {
                if (FAILED(q.queue->Signal(q.fence, q.signaled + 1))) {
                    queued = false;
                    break;
                }
                item.targets.push_back(++q.signaled);
            }
        }
        if (queued) {
            retired_resources.push_back(std::move(item));
            InterlockedIncrement(&retire_pending);
        }
    } catch (const std::bad_alloc&) {
        queued = false;
    }
    ReleaseSRWLockExclusive(&retire_lock);
    if (!queued) InterlockedIncrement64(&retire_kept);  // stays retained, as without the switch
}

// A null SRV of the dimension the title's view had (the draw path binds the same R8G8B8A8_UNORM null for a view that
// resolves to nothing). A null resource reads no more of the description than its format and dimension.
D3D12_SHADER_RESOURCE_VIEW_DESC retire_null_srv_desc(const ViewRecord& record)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    if (record.desc_bytes >= 40) {
        D3D12_SRV_DIMENSION dimension;
        std::memcpy(&dimension, record.desc + 4, sizeof(dimension));
        if (dimension >= D3D12_SRV_DIMENSION_TEXTURE1D && dimension <= D3D12_SRV_DIMENSION_TEXTURECUBEARRAY)
            desc.ViewDimension = dimension;
    }
    switch (desc.ViewDimension) {
    case D3D12_SRV_DIMENSION_TEXTURE1D: desc.Texture1D.MipLevels = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURE1DARRAY: desc.Texture1DArray.MipLevels = 1; desc.Texture1DArray.ArraySize = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURE2DARRAY: desc.Texture2DArray.MipLevels = 1; desc.Texture2DArray.ArraySize = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY: desc.Texture2DMSArray.ArraySize = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURE3D: desc.Texture3D.MipLevels = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURECUBE: desc.TextureCube.MipLevels = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY: desc.TextureCubeArray.MipLevels = 1; desc.TextureCubeArray.NumCubes = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURE2DMS: break;
    default: desc.Texture2D.MipLevels = 1; break;
    }
    return desc;
}

// True when the release-time fence of every queue has passed (retire_lock held).
bool retire_fences_reached(const RetiredResource& item)
{
    bool reached = item.targets.size() <= retire_queues.size();
    for (std::size_t q = 0; q < item.targets.size() && reached; ++q)
        reached = retire_queues[q].fence->GetCompletedValue() >= item.targets[q];
    return reached;
}

// X2W_D3D12_RELEASE_RETIRED_VIEWS=1: a released class member whose release-time fences have passed and that only
// tracked slot-18 descriptor slots still name gets those slots replaced by null descriptors, so the main drain loop can
// free it. Every slot of every heap that has a shadow is handled, copies included (they are counted the same way). For
// each slot, under note_lock: the shadow entry is cleared and its ref dropped, and the desktop descriptor is rewritten
// as a null SRV of the same dimension. The view records that name the resource then lose it (so a token that a raw
// title copy left behind resolves to nothing), and the token in title descriptor memory is cleared when it still
// carries the id. A slot that cannot be rewritten keeps its ref, and the resource stays retained. Nothing waits.
// Why this cannot reach GPU work: translated draws never bind a title heap. They build their own descriptors from the
// title's tokens at draw time and retire_hold what they sample, so a descriptor of a title heap is read by no recorded
// list or submission (ClearUnorderedAccessView binds a heap for UAV slots only, and those are not touched here).
void retire_null_ready_views()
{
    std::vector<Wrapper*> ready;
    try {
        ready.reserve(64);
    } catch (const std::bad_alloc&) {
        return;
    }
    if (!TryAcquireSRWLockShared(&retire_lock)) return;
    for (const RetiredResource& item : retired_resources) {
        Wrapper* w = item.wrapper;
        if (ready.size() == ready.capacity()) break;
        if (w->own().retire_hold_missed || w->own().retire_views_stuck || !retire_fences_reached(item)) continue;
        if (InterlockedCompareExchange(&w->own().retire_view_refs, 0, 0) <= 0) continue;
        if (InterlockedCompareExchange(&w->own().refs, 0, 0) != 0) continue;  // referenced again: leave its views
        ready.push_back(w);
    }
    ReleaseSRWLockShared(&retire_lock);
    if (ready.empty()) return;

    struct Cleared {
        u64 title_cpu;
        std::uint32_t index, increment, id;
    };
    std::vector<Cleared> cleared;
    std::size_t stuck = 0;
    // draw_lock first, in the order draw_translated takes them: a draw holds it from reading a view record until it
    // retire_holds the texture, so clearing the records under it means a draw either recorded its hold already or
    // resolves the token to nothing.
    if (!TryAcquireSRWLockExclusive(&draw_lock)) return;
    if (!TryAcquireSRWLockExclusive(&note_lock)) {
        ReleaseSRWLockExclusive(&draw_lock);
        return;
    }
    try {
        std::unordered_set<std::uint32_t> ids;  // view ids that name a ready resource
        for (std::uint32_t id = 0; id < view_records().size(); ++id) {
            const ViewRecord& record = view_records()[id];
            if (record.slot != 18 || !view_materialized()[id]) continue;
            if (std::find(ready.begin(), ready.end(), reinterpret_cast<Wrapper*>(record.resource)) != ready.end())
                ids.insert(id);
        }
        ID3D12Device* const device = the_device ? device_of(the_device) : nullptr;
        for (const HeapInfo& h : heap_infos()) {
            if (!h.shadow || ids.empty() || !device || !h.desktop_cpu) continue;
            for (std::uint32_t i = 0; i < h.count; ++i) {
                const std::uint32_t slot_id = h.shadow[i];
                if (!slot_id || !ids.count(slot_id - 1)) continue;
                const ViewRecord& record = view_records()[slot_id - 1];
                const D3D12_SHADER_RESOURCE_VIEW_DESC null_view = retire_null_srv_desc(record);
                device->CreateShaderResourceView(nullptr, &null_view,
                    D3D12_CPU_DESCRIPTOR_HANDLE{static_cast<SIZE_T>(h.desktop_cpu + u64{i} * h.increment)});
                h.shadow[i] = 0;
                retire_release_view_ref(reinterpret_cast<const void*>(record.resource));
                if (h.title_cpu && h.title_cpu != h.desktop_cpu) cleared.push_back({h.title_cpu, i, h.increment, slot_id - 1});
            }
        }
        for (Wrapper* w : ready) {
            if (InterlockedCompareExchange(&w->own().retire_view_refs, 0, 0) > 0) {
                w->own().retire_views_stuck = true;
                ++stuck;
                continue;
            }
            for (std::uint32_t id : ids)
                if (view_records()[id].resource == reinterpret_cast<u64>(w)) view_records()[id].resource = 0;
        }
    } catch (const std::bad_alloc&) {
    }
    ReleaseSRWLockExclusive(&note_lock);
    ReleaseSRWLockExclusive(&draw_lock);
    for (const Cleared& slot : cleared) {
        const u64 token = slot.title_cpu + u64{slot.index} * title_descriptor_stride(slot.title_cpu, slot.increment);
        if (!title_backed(token, 32) || !x2w_local::writable(reinterpret_cast<void*>(token), 32)) continue;
        std::uint32_t words[2];
        std::memcpy(words, reinterpret_cast<const void*>(token), sizeof(words));
        if (words[0] == view_token_magic && words[1] == slot.id)
            InterlockedCompareExchange(reinterpret_cast<volatile LONG*>(token), 0, static_cast<LONG>(view_token_magic));
    }
    if (stuck) {
        static LONG reported;
        if (first_calls(reported, 4)) log("x2w-d3d12 retire: %zu released textures keep an untracked descriptor; they stay retained", stuck);
    }
}


// Never blocks: a contended lock or an unfinished queue leaves the work for the next call.
void drain_retired_resources()
{
    if (!InterlockedCompareExchange(&retire_pending, 0, 0)) return;
    if (release_retired_views_enabled()) retire_null_ready_views();
    constexpr std::size_t batch = 256;
    struct RetiredDrop {
        Wrapper* wrapper;
        ID3D12Resource* resource;
    };
    std::vector<RetiredDrop> dropped;
    std::vector<std::vector<ID3D12Resource*>> dropped_flights;
    try {
        dropped.reserve(batch);
        dropped_flights.reserve(batch);
    } catch (const std::bad_alloc&) {
        return;
    }
    if (!TryAcquireSRWLockExclusive(&retire_lock)) return;
    for (std::size_t i = 0; i < retired_resources.size() && dropped.size() < batch;) {
        RetiredResource& item = retired_resources[i];
        Wrapper* w = item.wrapper;
        if (w->own().retire_hold_missed) {
            // retire_hold could not record a list's use of this resource: keep its desktop object for the process,
            // like a submission that could not be fenced, and drop only the retire bookkeeping.
            InterlockedIncrement64(&retire_kept);
            if (i + 1 != retired_resources.size()) item = std::move(retired_resources.back());
            retired_resources.pop_back();
            InterlockedDecrement(&retire_pending);
            continue;
        }
        if (InterlockedCompareExchange(&w->own().retire_view_refs, 0, 0) > 0) {
            ++i;
            continue;
        }
        bool reached = item.targets.size() <= retire_queues.size();
        for (std::size_t q = 0; q < item.targets.size() && reached; ++q)
            reached = retire_queues[q].fence->GetCompletedValue() >= item.targets[q];
        if (!reached) {
            ++i;
            continue;
        }
        // Claim the reference count so that a late AddRef cannot bring the wrapper back while its object goes.
        if (InterlockedCompareExchange(&w->own().refs, retired_refs, 0) == 0) {
            dropped.push_back({w, static_cast<ID3D12Resource*>(w->own().real)});
            w->own().real = nullptr;
        } else {
            InterlockedIncrement64(&retire_kept);  // referenced again after its final Release
        }
        if (i + 1 != retired_resources.size()) item = std::move(retired_resources.back());
        retired_resources.pop_back();
        InterlockedDecrement(&retire_pending);
    }
    for (std::size_t i = 0; i < retire_flights.size() && dropped_flights.size() < batch;) {
        RetireFlight& flight = retire_flights[i];
        if (retire_queues[flight.queue].fence->GetCompletedValue() < flight.value) {
            ++i;
            continue;
        }
        dropped_flights.push_back(std::move(flight.resources));
        if (i + 1 != retire_flights.size()) flight = std::move(retire_flights.back());
        retire_flights.pop_back();
        InterlockedDecrement(&retire_pending);
    }
    ReleaseSRWLockExclusive(&retire_lock);
    for (const RetiredDrop& drop : dropped) {
        if (drop.wrapper->own().title_memory) {
            const u64 address = reinterpret_cast<u64>(drop.wrapper->own().title_memory);
            {
                x2w_startup::ExclusiveNoteLock guard(note_lock);
                resource_address_index().erase(address, drop.wrapper);
            }
            if (drop.wrapper->own().retire_has_written_gen && !retire_address_has_live_user(address, drop.wrapper)) {
                AcquireSRWLockExclusive(&written_lock);
                if (const auto found = written_textures().find(address);
                    found != written_textures().end() && found->second.generation == drop.wrapper->own().retire_written_gen)
                    written_textures().erase(found);
                ReleaseSRWLockExclusive(&written_lock);
            }
        }
        AcquireSRWLockExclusive(&draw_lock);
        texture_uploads().erase(drop.resource);
        ReleaseSRWLockExclusive(&draw_lock);
        drop.resource->Release();
        const LONG64 total = InterlockedIncrement64(&retire_freed);
        if (total <= 8 || total % 1000 == 0) log("x2w-d3d12 retire: %lld desktop references dropped", total);
    }
    for (const auto& resources : dropped_flights) {
        for (ID3D12Resource* resource : resources) {
            resource->Release();
            const LONG64 total = InterlockedIncrement64(&retire_freed);
            if (total <= 8 || total % 1000 == 0) log("x2w-d3d12 retire: %lld desktop references dropped", total);
        }
    }
}

// X2W_D3D12_RETIRE_REASONS=1 (with X2W_D3D12_RELEASE_RETIRED=1 and X2W_D3D12_BUDGET=1): why resources that the guest
// released for good still hold their desktop object. Every final guest Release of a resource with a desktop object is
// remembered here; the heartbeat classifies the ones whose object is still alive, once each, by the first cause that
// applies, in the order of the enum below. Counts and bytes come from the budget ticket. Without the switches nothing
// is recorded and no line is built.
bool retire_reasons_enabled()
{
    static const bool on = release_retired_enabled() && x2w_budget::enabled() && x2w_local::env_on("X2W_D3D12_RETIRE_REASONS");
    return on;
}

struct ReleasedResource {
    ID3D12Resource* real;  // identity once the wrapper let go of it; dereferenced only while a hold keeps it alive
    LONG frame;            // frame_number at the final Release
};
SRWLOCK retire_diag_lock = SRWLOCK_INIT;  // a leaf, never held together with another lock
std::unordered_map<Wrapper*, ReleasedResource> retire_diag_released;

void retire_diag_note_release(Wrapper* w)
{
    if (!w->own().real) return;
    AcquireSRWLockExclusive(&retire_diag_lock);
    try {
        retire_diag_released[w] = {static_cast<ID3D12Resource*>(w->own().real), frame_number};
    } catch (const std::bad_alloc&) {  // a diagnostic only: the resource is then missing from the counts
    }
    ReleaseSRWLockExclusive(&retire_diag_lock);
}

enum class Held { view_refs, list_hold, flight, waiting_fence, drain_pending, unqueued, left_view, left_view_copy,
                  left_residency, left_present, left_hold_missed, never_slot, never_buffer, never_dimension, never_msaa,
                  never_rt_ds_uav, never_heap_type, never_heap_flags, never_title_memory, never_aliased, never_reserved,
                  other, count };
constexpr const char* held_names[] = {"view-refs", "list-hold", "flight", "queued-waiting-fence", "drain-pending",
    "unqueued", "left-class:view", "left-class:view-copy", "left-class:residency", "left-class:present",
    "left-class:hold-missed", "never-class:slot", "never-class:buffer", "never-class:dimension", "never-class:msaa",
    "never-class:rt-ds-uav", "never-class:heap-type", "never-class:heap-flags", "never-class:title-memory",
    "never-class:aliased", "never-class:reserved", "other"};
static_assert(std::size(held_names) == static_cast<std::size_t>(Held::count));
constexpr Held held_left[] = {Held::other, Held::left_view, Held::left_view_copy, Held::left_residency, Held::left_present,
                              Held::left_hold_missed};
constexpr Held held_never[] = {Held::never_slot, Held::other, Held::never_buffer, Held::never_dimension, Held::never_msaa,
                               Held::never_rt_ds_uav, Held::never_heap_type, Held::never_heap_flags,
                               Held::never_title_memory, Held::never_aliased, Held::never_reserved};
static_assert(std::size(held_left) == static_cast<std::size_t>(RetireLeft::count));
static_assert(std::size(held_never) == static_cast<std::size_t>(RetireNever::count));

struct RetireReasonReport {
    u64 count[static_cast<int>(Held::count)];
    u64 bytes[static_cast<int>(Held::count)];
    LONG oldest[static_cast<int>(Held::count)];  // frames since the final Release of the oldest entry
    u64 visible_count, visible_bytes;            // view-refs entries with a descriptor in a shader-visible heap
};

void retire_reasons_collect(RetireReasonReport& report)
{
    // 1: which resources a tracked slot-18 descriptor names, and whether one of those lies in a shader-visible heap.
    std::unordered_map<const Wrapper*, bool> viewed;
    AcquireSRWLockShared(&note_lock);
    try {
        for (const HeapInfo& h : heap_infos()) {
            if (!h.shadow) continue;
            for (std::uint32_t i = 0; i < h.count; ++i) {
                const std::uint32_t id = h.shadow[i];
                if (!id || id - 1 >= view_records().size() || view_records()[id - 1].slot != 18 || !view_materialized()[id - 1])
                    continue;
                bool& visible = viewed[reinterpret_cast<const Wrapper*>(view_records()[id - 1].resource)];
                visible = visible || h.desktop_gpu != 0;
            }
        }
    } catch (const std::bad_alloc&) {
    }
    ReleaseSRWLockShared(&note_lock);

    // 2: the remembered final Releases.
    std::vector<std::pair<Wrapper*, ReleasedResource>> entries;
    AcquireSRWLockShared(&retire_diag_lock);
    try {
        entries.assign(retire_diag_released.begin(), retire_diag_released.end());
    } catch (const std::bad_alloc&) {
    }
    ReleaseSRWLockShared(&retire_diag_lock);

    // 3: the retire bookkeeping. own().real is cleared by the drain under this lock, so a non-null pointer read here
    // names a desktop object that stays alive for as long as the lock is held.
    std::vector<Wrapper*> gone;
    AcquireSRWLockShared(&retire_lock);
    try {
        std::unordered_map<const Wrapper*, const RetiredResource*> queued;
        std::unordered_set<const ID3D12Resource*> listed, flying;
        for (const RetiredResource& item : retired_resources) queued.emplace(item.wrapper, &item);
        for (const auto& list : retire_list_holds) listed.insert(list.second.begin(), list.second.end());
        for (const RetireFlight& flight : retire_flights) flying.insert(flight.resources.begin(), flight.resources.end());
        const LONG now = frame_number;
        const auto add = [&](Held why, const x2w_budget::Ticket* ticket, LONG frame) {
            const int i = static_cast<int>(why);
            ++report.count[i];
            report.bytes[i] += ticket ? static_cast<u64>(ticket->bytes()) : 0;
            report.oldest[i] = (std::max)(report.oldest[i], now - frame);
        };
        for (const auto& [w, released] : entries) {
            if (InterlockedCompareExchange(&w->own().refs, 0, 0) > 0) continue;  // referenced again: not released for good
            auto* real = static_cast<ID3D12Resource*>(w->own().real);
            if (!real) {
                // The drain let go of it; only a list hold or a submission can still keep the desktop object.
                const bool by_list = listed.count(released.real) != 0;
                if (!by_list && !flying.count(released.real)) {
                    gone.push_back(w);
                    continue;
                }
                x2w_budget::Ticket* ticket = x2w_budget::ticket_of(released.real);
                add(by_list ? Held::list_hold : Held::flight, ticket, released.frame);
                if (ticket) ticket->Release();
                continue;
            }
            const x2w_budget::Ticket* ticket = w->own().budget;
            Held why = Held::other;
            const auto found = queued.find(w);
            if (!w->own().retire_when_idle) {
                if (w->own().retire_left && w->own().retire_left < static_cast<std::uint8_t>(RetireLeft::count))
                    why = held_left[w->own().retire_left];
                else if (w->own().retire_never < static_cast<std::uint8_t>(RetireNever::count))
                    why = held_never[w->own().retire_never];
            } else if (found == queued.end()) {
                why = Held::unqueued;
            } else if (InterlockedCompareExchange(&w->own().retire_view_refs, 0, 0) > 0) {
                why = Held::view_refs;
                if (const auto view = viewed.find(w); view != viewed.end() && view->second) {
                    ++report.visible_count;
                    report.visible_bytes += ticket ? static_cast<u64>(ticket->bytes()) : 0;
                }
            } else if (listed.count(real)) {
                why = Held::list_hold;
            } else if (flying.count(real)) {
                why = Held::flight;
            } else {
                const RetiredResource& item = *found->second;
                bool reached = item.targets.size() <= retire_queues.size();
                for (std::size_t q = 0; q < item.targets.size() && reached; ++q)
                    reached = retire_queues[q].fence->GetCompletedValue() >= item.targets[q];
                why = reached ? Held::drain_pending : Held::waiting_fence;
            }
            add(why, ticket, released.frame);
        }
    } catch (const std::bad_alloc&) {
    }
    ReleaseSRWLockShared(&retire_lock);

    // 4: forget entries whose desktop object is gone.
    if (!gone.empty()) {
        AcquireSRWLockExclusive(&retire_diag_lock);
        for (Wrapper* w : gone)
            if (!w->own().real) retire_diag_released.erase(w);
        ReleaseSRWLockExclusive(&retire_diag_lock);
    }
}

// The heartbeat line, or an empty string when the diagnostics are not switched on. unattributed: what the budget still
// counts as retained, less what this classification found (backing heaps, upload and readback objects, anything it cannot see).
std::size_t retire_reasons_line(char* out, std::size_t capacity)
{
    if (!out || !capacity) return 0;
    out[0] = 0;
    if (!retire_reasons_enabled()) return 0;
    RetireReasonReport report{};
    retire_reasons_collect(report);
    const x2w_budget::Snapshot budget = x2w_budget::snapshot();
    std::size_t used = 0;
    const auto append = [&](const char* format, auto... args) {
        if (used + 1 >= capacity) return;
        const int n = std::snprintf(out + used, capacity - used, format, args...);
        if (n > 0) used = (std::min)(used + static_cast<std::size_t>(n), capacity - 1);
    };
    append("x2w-retire held:");
    u64 total_count = 0, total_bytes = 0;
    for (int i = 0; i < static_cast<int>(Held::count); ++i) {
        if (!report.count[i]) continue;
        total_count += report.count[i];
        total_bytes += report.bytes[i];
        append(" %s=%llu/%llu(oldest %ldf)", held_names[i], report.count[i], report.bytes[i], report.oldest[i]);
    }
    if (!total_count) append(" none");
    if (report.visible_count) append(" view-refs-in-visible-heap=%llu/%llu", report.visible_count, report.visible_bytes);
    u64 retained_count = 0, retained_bytes = 0;
    for (const auto& retained : budget.retained) {
        retained_count += retained.objects;
        retained_bytes += retained.bytes;
    }
    append(" total=%llu/%llu unattributed=%lld/%lld", total_count, total_bytes,
           static_cast<long long>(retained_count) - static_cast<long long>(total_count),
           static_cast<long long>(retained_bytes) - static_cast<long long>(total_bytes));
    return used;
}

// Fixture export (startup_resource_retire_reasons): the line the heartbeat logs, now. Returns its length; 0 and an
// empty string without the switches.
extern "C" __declspec(dllexport) std::size_t x2w_d3d12_retire_reasons_line(char* out, std::size_t capacity)
{
    return retire_reasons_line(out, capacity);
}

// Fixture probe (startup_resource_retire): what still names a resource of the class by its raw desktop pointer, how many
// resources the open recording of a list moved out of COMMON, and how many it holds. Counts entries only.
struct RetireProbe {
    unsigned cache_entries;
    unsigned list_states;
    unsigned list_holds;
};
extern "C" __declspec(dllexport) BOOL x2w_d3d12_retire_probe(void* resource, void* list, RetireProbe* out)
{
    if (!out) return FALSE;
    *out = {};
    if (!release_retired_enabled()) return FALSE;  // like x2w_d3d12_budget_snapshot: nothing to report without the switch
    if (Wrapper* r = as_wrapper(resource); r && r->own().kind == resource_kind && r->own().real) {
        auto* real = static_cast<ID3D12Resource*>(r->own().real);
        AcquireSRWLockExclusive(&draw_lock);
        out->cache_entries += static_cast<unsigned>(texture_uploads().count(real));
        ReleaseSRWLockExclusive(&draw_lock);
        AcquireSRWLockShared(&target_lock);
        for (const auto& view : view_targets()) out->cache_entries += view.second.resource == real;
        ReleaseSRWLockShared(&target_lock);
        AcquireSRWLockShared(&pending_writes_lock);
        for (const auto& writes : pending_gpu_writes())
            out->cache_entries += static_cast<unsigned>(std::count(writes.second.begin(), writes.second.end(), real));
        ReleaseSRWLockShared(&pending_writes_lock);
    }
    if (Wrapper* l = as_wrapper(list); l && l->own().kind == list_kind) {
        if (l->own().pass) out->list_states = static_cast<unsigned>(pass_of(l).states.size());
        AcquireSRWLockShared(&retire_lock);
        if (const auto found = retire_list_holds.find(l); found != retire_list_holds.end())
            out->list_holds = static_cast<unsigned>(found->second.size());
        ReleaseSRWLockShared(&retire_lock);
    }
    return TRUE;
}

// Fixture probe (startup_teardown_unload, #2572) for the two caches that keep desktop objects for a
// device: the indexed-validation pipeline (d3d12_draw.inl, bit 0) and the rectangle-list argument
// converter (rect_indirect_args.h, bit 1). Warms the selected caches the way their first real call
// would -- the validation pipeline directly, the converter through one record-only conversion, nothing
// executed -- and answers how many objects they now hold for `device`: the validation entry's device,
// root signature and pipeline (3), plus the converter entry's root signature and pipeline (2). The
// caller can warm one cache at a time to see each one release what it owns when the module is unloaded.
extern "C" __declspec(dllexport) UINT x2w_d3d12_cached_objects_probe(ID3D12Device* device, UINT caches)
{
    if (!device || !caches || (caches & ~3u)) return 0;
    UINT owned = 0;
    if (caches & 1u) {
        const auto& validation = indexed_validation_pipeline(device);
        if (validation.device) ++owned;
        if (validation.root) ++owned;
        if (validation.pipeline) ++owned;
    }
    if (!(caches & 2u)) return owned;

    x2w_startup::ComPtr<ID3D12CommandAllocator> allocator;
    x2w_startup::ComPtr<ID3D12GraphicsCommandList> list;
    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                        IID_PPV_ARGS(&list))))
        return owned;
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = 256;
    buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    // convert() copies its 16 source words; an uninitialised buffer in the state it requires is enough.
    x2w_startup::ComPtr<ID3D12Resource> source;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                               nullptr, IID_PPV_ARGS(&source))))
        return owned;
    const auto scratch = x2w_startup::RectIndirectArguments::make_scratch(device);
    if (scratch && x2w_startup::rect_indirect_arguments().convert(list.Get(), source.Get(), 0, scratch.Get()))
        owned += 2;  // the converter's root signature and pipeline for this device
    return owned;
}

// Fixture probes (startup_resource_retire_title) for state the retire bookkeeping maintains for one resource. They read
// the same maps the shim reads and never change them; the address index and the upload cache exist with and without
// the switch, so these answer in both modes.
extern "C" __declspec(dllexport) BOOL x2w_d3d12_resource_index_probe(void* resource)
{
    Wrapper* w = as_wrapper(resource);
    if (!w || w->own().kind != resource_kind) return FALSE;
    const u64 base = reinterpret_cast<u64>(w->own().title_memory);
    if (!base) return FALSE;
    AcquireSRWLockShared(&note_lock);
    const bool indexed = resource_address_index().contains(base, static_cast<const Wrapper*>(w));
    ReleaseSRWLockShared(&note_lock);
    return indexed ? TRUE : FALSE;
}

// The desktop pointer is not dereferenced: the fixture keeps its own desktop reference to a retired resource.
extern "C" __declspec(dllexport) BOOL x2w_d3d12_texture_upload_probe(void* desktop_resource)
{
    if (!desktop_resource) return FALSE;
    AcquireSRWLockExclusive(&draw_lock);
    const bool cached = texture_uploads().count(static_cast<ID3D12Resource*>(desktop_resource)) != 0;
    ReleaseSRWLockExclusive(&draw_lock);
    return cached ? TRUE : FALSE;
}

extern "C" __declspec(dllexport) BOOL x2w_d3d12_written_probe(void* address, unsigned long long* generation, unsigned* bc7_width,
                                                            unsigned* desktop_rows)
{
    if (!address) return FALSE;
    AcquireSRWLockShared(&written_lock);
    const auto found = written_textures().find(reinterpret_cast<u64>(address));
    const bool present = found != written_textures().end();
    if (present) {
        if (generation) *generation = found->second.generation;
        if (bc7_width) *bc7_width = found->second.bc7_width;
        if (desktop_rows) *desktop_rows = found->second.desktop_rows ? 1u : 0u;
    }
    ReleaseSRWLockShared(&written_lock);
    return present ? TRUE : FALSE;
}

// The title exports __XMemAllocationHooks__ = {alloc(size, attributes), free(ptr, attributes), ...}.
// Its alloc routine selects a hook chain by allocatorId (attribute bits 16-23). Around the creation
// of its two display render targets the title pushes a bump allocator for allocatorId 0x80 that
// serves pageSize 2 (bits 14-15) from a 64 MiB block, then asserts that the block was used up:
// the console runtime evidently takes committed-resource memory from XMemAlloc with allocatorId
// 0x80. Hypothesis used here: attributes = pageSize (2 from 4 MiB up, else 1) << 14 | 0x80 << 16 |
// log2(alignment) << 24, memory type 0. The block is bookkeeping only; the desktop resource has its
// own memory and does not alias it.
using XMemAllocHook = void*(__stdcall*)(SIZE_T, ULONGLONG);
XMemAllocHook title_alloc_hook()
{
    static const XMemAllocHook hook = [] {
        auto* const* table = reinterpret_cast<void* const*>(GetProcAddress(GetModuleHandleW(nullptr), "__XMemAllocationHooks__"));
        return table ? reinterpret_cast<XMemAllocHook>(table[0]) : nullptr;
    }();
    return hook;
}

void* title_resource_memory(UINT64 size, UINT64 alignment)
{
    const XMemAllocHook hook = title_alloc_hook();
    if (!hook || !size) return nullptr;
    unsigned log2_alignment = 0;
    while ((UINT64{1} << (log2_alignment + 1)) <= alignment && log2_alignment < 31) ++log2_alignment;
    const ULONGLONG page = size >= (UINT64{4} << 20) ? 2 : 1;
    const ULONGLONG attributes = (page << 14) | (ULONGLONG{0x80} << 16) | (ULONGLONG{log2_alignment} << 24);
    void* memory = hook(static_cast<SIZE_T>(size), attributes);
    static LONG calls;
    if (first_calls(calls, 12))
        log("x2w-d3d12 title XMem hook alloc size=0x%llx attributes=0x%llx -> %p", size, attributes, memory);
    return memory;
}

// X2W_D3D12_DUMP_PSO=1: the first pipeline descriptions of each creation slot are written to the
// log: the description itself and, for every qword in it that points at readable memory, the first
// bytes there (shader containers start with "DXBC"). Diagnostics for the translation work.
void dump_pipeline(const char* what, int slot, const void* desc, std::size_t bytes)
{
    static const bool enabled = [] {
        char value[4]{};
        return GetEnvironmentVariableA("X2W_D3D12_DUMP_PSO", value, sizeof(value)) && value[0] == '1';
    }();
    static LONG dumped[512];
    if (!enabled || !desc || !x2w_local::first_calls(dumped[slot & 511], 4)) return;
    if (!x2w_local::readable(desc, bytes)) return;
    const auto* words = static_cast<const u64*>(desc);
    log("x2w-pso %s slot %d desc %p, %llu bytes", what, slot, desc, static_cast<unsigned long long>(bytes));
    for (std::size_t i = 0; i < bytes / 8; i += 4)
        log("x2w-pso   +0x%03llx: %016llx %016llx %016llx %016llx", static_cast<unsigned long long>(i * 8), words[i],
            i + 1 < bytes / 8 ? words[i + 1] : 0, i + 2 < bytes / 8 ? words[i + 2] : 0, i + 3 < bytes / 8 ? words[i + 3] : 0);
    for (std::size_t i = 0; i < bytes / 8; ++i) {
        const auto* target = reinterpret_cast<const unsigned char*>(words[i]);
        if (words[i] < 0x10000 || as_wrapper(target) || !x2w_local::readable(target, 64)) continue;
        char hex[3 * 48 + 1], text[49];
        hex_text(target, 48, hex, sizeof(hex));
        for (int c = 0; c < 48; ++c) text[c] = target[c] >= 0x20 && target[c] < 0x7f ? static_cast<char>(target[c]) : '.';
        text[48] = 0;
        const u64 next = i + 1 < bytes / 8 ? words[i + 1] : 0;
        log("x2w-pso   [+0x%03llx] -> %p (next word 0x%llx) %s |%s|", static_cast<unsigned long long>(i * 8),
            static_cast<const void*>(target), next, hex, text);
    }
}

// X2W_SHADER_DUMP_DIR=<dir>: every DXBC container a pipeline description points at is written to
// <dir>/<digest>.dxbc (digest = the 16 bytes after the "DXBC" tag), and each stream description to
// <dir>/stream_<n>.bin with a .txt listing which stream offsets point at which container. These feed
// the offline retarget/sign step (tools/dxil_retarget.py); translated results are loaded back by digest.
bool shader_dump_dir(wchar_t* out, DWORD size)
{
    const DWORD n = GetEnvironmentVariableW(L"X2W_SHADER_DUMP_DIR", out, size);
    return n && n < size;
}

void write_file(const wchar_t* path, const void* data, std::size_t bytes)
{
    HANDLE file = CreateFileW(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return;
    DWORD written = 0;
    WriteFile(file, data, static_cast<DWORD>(bytes), &written, nullptr);
    CloseHandle(file);
}

bool is_container(const void* p)
{
    return p && x2w_local::readable(p, 32) && std::memcmp(p, "DXBC", 4) == 0;
}

void digest_text(const void* container, wchar_t* out)
{
    const auto* b = static_cast<const unsigned char*>(container) + 4;
    for (int i = 0; i < 16; ++i) swprintf(out + 2 * i, 3, L"%02x", b[i]);
}

void dump_containers(const char* what, const void* desc, std::size_t bytes)
{
    wchar_t dir[MAX_PATH];
    if (!desc || !shader_dump_dir(dir, MAX_PATH) || !x2w_local::readable(desc, bytes)) return;
    static LONG streams;
    const LONG index = InterlockedIncrement(&streams);
    char listing[4096];
    int used = std::snprintf(listing, sizeof(listing), "%s description, %llu bytes\n", what,
                             static_cast<unsigned long long>(bytes));
    const auto* words = static_cast<const u64*>(desc);
    for (std::size_t i = 0; i < bytes / 8; ++i) {
        const void* target = reinterpret_cast<const void*>(words[i]);
        if (words[i] < 0x10000 || !is_container(target)) continue;
        const std::uint32_t total = *reinterpret_cast<const std::uint32_t*>(static_cast<const char*>(target) + 24);
        if (total < 32 || total > (64u << 20) || !x2w_local::readable(target, total)) continue;
        wchar_t digest[33], path[MAX_PATH];
        digest_text(target, digest);
        swprintf(path, MAX_PATH, L"%s\\%s.dxbc", dir, digest);
        if (GetFileAttributesW(path) == INVALID_FILE_ATTRIBUTES) write_file(path, target, total);
        if (used < static_cast<int>(sizeof(listing)) - 96)
            used += std::snprintf(listing + used, sizeof(listing) - used, "+0x%03llx %ls %u\n",
                                  static_cast<unsigned long long>(i * 8), digest, total);
    }
    wchar_t path[MAX_PATH];
    swprintf(path, MAX_PATH, L"%s\\%s_%04ld.bin", dir, what[0] == 's' ? L"stream" : L"compute", index);
    write_file(path, desc, bytes);
    swprintf(path, MAX_PATH, L"%s\\%s_%04ld.txt", dir, what[0] == 's' ? L"stream" : L"compute", index);
    write_file(path, listing, static_cast<std::size_t>(used));
}

// Placeholder objects (no desktop object behind the wrapper): ID3D12Object methods succeed and do nothing.
HRESULT __stdcall placeholder_ok(Wrapper*, u64, u64, u64) { return S_OK; }

// Device slot 62 (Xbox order): (D3D12_COMPUTE_PIPELINE_STATE_DESC*, 0, 0, riid, ppv); first seen from
// XboxBvhBuilder.dll, followed by SetName on the result. The bytecode is console GPU code, not DXIL,
// so no desktop pipeline is built: the result is a placeholder PipelineState until shader translation
// exists. Logs the root signature, bytecode address, length and first 32 bytes.
HRESULT __stdcall device_create_compute_pipeline_x(Wrapper*, const D3D12_COMPUTE_PIPELINE_STATE_DESC* desc, u64 extra0,
                                                   u64 extra1, const GUID* iid, void** out)
{
    char text[48];
    guid_text(iid, text, sizeof(text));
    if (!out || !desc || !iid) return E_POINTER;
    *out = nullptr;
    dump_pipeline("compute", 62, desc, sizeof(*desc));
    dump_containers("compute", desc, sizeof(*desc));
    char bytes[3 * 32 + 1];
    hex_text(desc->CS.pShaderBytecode, desc->CS.BytecodeLength < 32 ? desc->CS.BytecodeLength : 32, bytes, sizeof(bytes));
    static LONG calls;
    if (first_calls(calls, 24))
        log("x2w-d3d12 Device::slot62(CreateComputePipelineState?) root=%p cs=%p len=0x%llx node=0x%x flags=0x%x extra=0x%llx,0x%llx "
            "riid=%s -> placeholder [%s]", static_cast<void*>(desc->pRootSignature), desc->CS.pShaderBytecode,
            static_cast<unsigned long long>(desc->CS.BytecodeLength), desc->NodeMask, desc->Flags, extra0, extra1, text, bytes);
    // LOCAL: write the compute code and its root signature to the shader cache as req_cs_<key>.{cs,root}
    // for the offline compute translator (nothing executes dispatches yet).
    u64 compute_key = 0;
    if (Wrapper* root = as_wrapper(desc->pRootSignature); root && desc->CS.BytecodeLength &&
        desc->CS.BytecodeLength <= (1u << 20) && x2w_local::readable(desc->CS.pShaderBytecode, desc->CS.BytecodeLength)) {
        std::vector<unsigned char> root_blob;
        AcquireSRWLockShared(&note_lock);
        const auto found = root_blobs().find(root);
        if (found != root_blobs().end()) root_blob = found->second;
        ReleaseSRWLockShared(&note_lock);
        const u64 key = fnv_bytes(root_blob.data(), root_blob.size(),
                                  fnv_bytes(desc->CS.pShaderBytecode, desc->CS.BytecodeLength));
        if (!root_blob.empty()) compute_key = key;
        wchar_t path[MAX_PATH];
        if (!root_blob.empty() && cache_path(path, L"req_cs_", key, L".cs") && GetFileAttributesW(path) == INVALID_FILE_ATTRIBUTES) {
            write_file(path, desc->CS.pShaderBytecode, desc->CS.BytecodeLength);
            cache_path(path, L"req_cs_", key, L".root");
            write_file(path, root_blob.data(), root_blob.size());
        }
    }
    if (kind_for(*iid) != pipeline_kind) return E_NOINTERFACE;
    *out = wrap(nullptr, pipeline_kind);
    auto* pipeline = static_cast<Wrapper*>(*out);
    note_pipeline(pipeline, 62, nullptr, desc, sizeof(*desc));
    note_compute_binding_pipeline(pipeline, *desc);
    // LOCAL: a retargeted and signed container <key>.cs.dxbc becomes a desktop compute pipeline.
    if (compute_key) {
        wchar_t path[MAX_PATH];
        ID3DBlob* code = nullptr;
        Wrapper* root = as_wrapper(desc->pRootSignature);
        if (!rawaddr_pipeline(compute_key, root, static_cast<Wrapper*>(*out)) &&  // LOCAL: a sidecar key skips the plain path
            root && root->own().real && cache_path(path, L"", compute_key, L".cs.dxbc") && SUCCEEDED(D3DReadFileToBlob(path, &code))) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC real{};
            real.pRootSignature = static_cast<ID3D12RootSignature*>(root->own().real);
            real.CS = {code->GetBufferPointer(), code->GetBufferSize()};
            ID3D12PipelineState* pso = nullptr;
            const HRESULT hr = device_of(the_device)->CreateComputePipelineState(&real, IID_PPV_ARGS(&pso));
            log("x2w-compute pipeline %016llx from signed container -> 0x%08lx", compute_key, static_cast<unsigned long>(hr));
            if (SUCCEEDED(hr)) {
                AcquireSRWLockExclusive(&compute_pso_lock);
                compute_psos()[static_cast<Wrapper*>(*out)] = pso;
                ReleaseSRWLockExclusive(&compute_pso_lock);
            }
            code->Release();
        }
    }
    AcquireSRWLockExclusive(&compute_lock);
    compute_keys()[static_cast<Wrapper*>(*out)] = compute_key;
    ReleaseSRWLockExclusive(&compute_lock);
    return S_OK;
}

// Device slot 86 (Xbox order): (D3D12_PIPELINE_STATE_STREAM_DESC*, riid, ppv), the desktop
// CreatePipelineState shape (stream of 0x2b0 bytes first seen, ID3D12PipelineState IID). The stream
// carries console shaders; the result is a placeholder PipelineState.
HRESULT __stdcall device_create_pipeline_stream_x(Wrapper*, const D3D12_PIPELINE_STATE_STREAM_DESC* desc, const GUID* iid,
                                                  void** out)
{
    char text[48];
    guid_text(iid, text, sizeof(text));
    if (!out || !desc || !iid) return E_POINTER;
    *out = nullptr;
    dump_pipeline("stream", 86, desc->pPipelineStateSubobjectStream, desc->SizeInBytes < 0x400 ? desc->SizeInBytes : 0x400);
    dump_containers("stream", desc->pPipelineStateSubobjectStream, desc->SizeInBytes < 0x1000 ? desc->SizeInBytes : 0x1000);
    static LONG calls;
    if (first_calls(calls, 8))
        log("x2w-d3d12 Device::slot86(CreatePipelineState?) stream=%p bytes=0x%llx riid=%s",
            desc->pPipelineStateSubobjectStream, static_cast<unsigned long long>(desc->SizeInBytes), text);
    if (kind_for(*iid) != pipeline_kind) return E_NOINTERFACE;
    ID3D12PipelineState* real_pipeline = nullptr;
    Translation* translation = nullptr;
    D3D12_BLEND_DESC blend{};
    bool has_blend = false;
    D3D12_DEPTH_STENCIL_DESC depth_stencil{};
    bool has_depth_stencil = false;
    DXGI_FORMAT dsv_format = DXGI_FORMAT_UNKNOWN;
    D3D12_RASTERIZER_DESC rasterizer{};
    bool has_rasterizer = false;
    bool compute = false;
    IndirectGraphicsPipeline graphics;
    u64 mesh86_key = 0;
    if (render_mode()) {
        char why[256] = "";
        // Recognized desktop stages must reach the native path before the
        // guest cache creates a pending translation for any DXBC container.
        real_pipeline = build_stream_pipeline(desc, why, sizeof(why), compute, graphics);
        bool mesh_cache_request = false;
        // LOCAL merge: upstream's slot-86 mesh stage cache (#2270, req_<key>.m86) takes mesh streams before
        // stream_translation, which the local mesh path (X2W_MESH*) translates; it runs only with
        // X2W_UPSTREAM_MESH86_CACHE=1 so the local mesh path stays as before the merge.
        static const bool upstream_mesh86_cache = x2w_local::env_on("X2W_UPSTREAM_MESH86_CACHE");
        if (!real_pipeline && upstream_mesh86_cache)
            real_pipeline = try_cached_mesh86(*desc, why, sizeof(why), graphics, mesh_cache_request, mesh86_key);
        if (!mesh_cache_request && (!real_pipeline || graphics.root))
            translation = stream_translation(*desc, blend, has_blend, depth_stencil, has_depth_stencil, dsv_format,
                                             rasterizer, has_rasterizer);
        static LONG built;
        if ((real_pipeline || !translation) && first_calls(built, 24))
            log("x2w-render stream pipeline -> %s%s", real_pipeline ? "REAL desktop pipeline" : "placeholder: ", real_pipeline ? "" : why);
    }
    *out = wrap(real_pipeline, pipeline_kind);
    static_cast<Wrapper*>(*out)->own().compute_pipeline = real_pipeline && compute;
    static_cast<Wrapper*>(*out)->own().indirect_graphics = graphics;
    static_cast<Wrapper*>(*out)->own().mesh86_key = mesh86_key;
    note_pipeline(static_cast<Wrapper*>(*out), 86, nullptr, desc->pPipelineStateSubobjectStream, desc->SizeInBytes < 0x2000 ? desc->SizeInBytes : 0x2000,
                  translation, translation && has_blend ? &blend : nullptr,
                  translation && has_depth_stencil ? &depth_stencil : nullptr,
                  translation ? dsv_format : DXGI_FORMAT_UNKNOWN,
                  translation && has_rasterizer ? &rasterizer : nullptr);
    return S_OK;
}

// The fixed prefix shared with request_slot57_mesh. Only apply the creation
// refusal to the observed bounded spans, with a registered root and no VS.
bool is_fixed_mesh57_descriptor(const void* desc)
{
    u64 words[15];
    if (!x2w_local::readable(desc, sizeof(words)) || !copy_title_bytes(words, desc, sizeof(words))) return false;
    const auto readable_stage = [](u64 address, u64 bytes) {
        return bytes >= 16 && bytes <= 0x10000 &&
            x2w_local::readable(reinterpret_cast<const void*>(address), static_cast<std::size_t>(bytes));
    };
    const auto* root = as_wrapper(reinterpret_cast<const void*>(words[0]));
    return root && root->own().kind == root_signature_kind &&
        !words[9] && !words[10] &&
        readable_stage(words[13], words[14]) &&
        ((!words[5] && !words[6]) || readable_stage(words[5], words[6]));
}

// Device slots 57 and 67 (Xbox order): (desc*, riid, ppv) with the pipeline
// IID. These normally return a placeholder. In render mode slot 67 also
// publishes a compute translation request (d3d12_draw.inl).
template <int N>
HRESULT __stdcall device_create_pipeline_placeholder(Wrapper*, const void* desc, const GUID* iid, void** out)
{
    Timed timed(time_pipeline);
    char text[48];
    guid_text(iid, text, sizeof(text));
    if (!out || !iid) return E_POINTER;
    *out = nullptr;
    dump_pipeline("pipeline", N, desc, 0x300);
    if (kind_for(*iid) != pipeline_kind) return E_NOINTERFACE;
    if constexpr (N == 57) {
        if (x2w_local::env_on("X2W_REFUSE_FIXED_MESH") && is_fixed_mesh57_descriptor(desc)) {
            static LONG refusals;
            if (first_calls(refusals, 6))
                log("x2w-d3d12 Device::slot57: fixed mesh pipeline unavailable (E_NOTIMPL)");
            return E_NOTIMPL;
        }
    }
    static LONG calls;
    if (first_calls(calls, 2)) {
        char bytes[3 * 96 + 1];
        hex_text(desc, 96, bytes, sizeof(bytes));
        log("x2w-d3d12 Device::slot%d(pipeline?) desc=[%s] riid=%s -> placeholder", N, bytes, text);
    }
    if constexpr (N == 67) publish_compute_request(desc);
    *out = wrap(nullptr, pipeline_kind);
    note_pipeline(static_cast<Wrapper*>(*out), N, nullptr, desc, 0x300);
    if (N == 57) regblob::dump57(desc, static_cast<Wrapper*>(*out));
    if (N == 57) mesh_note57(desc, static_cast<Wrapper*>(*out));  // LOCAL meshrt: mesh key at creation (X2W_MESH=1)
    if (N == 67 &&x2w_local::readable(desc, 0x28)) {
        u64 words[5];
        std::memcpy(words, desc, sizeof(words));
        std::vector<unsigned char> regs;
        for (int i : {1, 3}) {
            const std::uint32_t bytes = words[i + 1] <= 0x1000 && x2w_local::readable(reinterpret_cast<const void*>(words[i]), words[i + 1])
                                            ? static_cast<std::uint32_t>(words[i + 1]) : 0u;
            regs.insert(regs.end(), reinterpret_cast<const unsigned char*>(&bytes), reinterpret_cast<const unsigned char*>(&bytes) + 4);
            if (bytes) regs.insert(regs.end(), reinterpret_cast<const unsigned char*>(words[i]), reinterpret_cast<const unsigned char*>(words[i]) + bytes);
        }
        AcquireSRWLockExclusive(&compute_lock);
        compute_regs()[static_cast<Wrapper*>(*out)] = std::move(regs);
        ReleaseSRWLockExclusive(&compute_lock);
        // LOCAL stall: key it now and load its translation in the background (d3d12_compute.inl).
        if (prewarm_enabled()) prewarm_compute(compute_key_for(static_cast<Wrapper*>(*out)));
    }
    return S_OK;
}

// CreateDerivedPipelineStateX: base, number of fixed-size state patches, patches, IID, result.
HRESULT __stdcall device_create_derived_pipeline(Wrapper*, void* base, UINT count, const void* desc, const GUID* iid,
                                                 void** out)
{
    Timed timed(time_pipeline);
    char text[48];
    guid_text(iid, text, sizeof(text));
    if (!out || !iid) return E_POINTER;
    *out = nullptr;
    if (kind_for(*iid) != pipeline_kind) return E_NOINTERFACE;
    if (count > 64) return E_INVALIDARG;
    const std::size_t bytes_in_patches = count * x2w_derived_blend::record_size;
    if (bytes_in_patches && (!desc || !x2w_local::readable(desc, bytes_in_patches))) return E_INVALIDARG;
    dump_pipeline("derived", 55, desc, bytes_in_patches);
    static LONG calls;
    if (first_calls(calls, 3)) {
        Wrapper* w = as_wrapper(base);
        char bytes[3 * 64 + 1];
        hex_text(desc, count ? 64 : 0, bytes, sizeof(bytes));
        log("x2w-d3d12 Device::slot55(derived pipeline) base=%p (%s) count=%u desc=[%s] riid=%s -> placeholder", base,
            w ? kind_names[w->own().kind] : "not a wrapper", count, bytes, text);
    }
    *out = wrap(nullptr, pipeline_kind);
    note_pipeline(static_cast<Wrapper*>(*out), 55, base, desc, bytes_in_patches);
    regblob::dump55(base, desc, bytes_in_patches);
    return S_OK;
}

// Device slot 127 (Xbox order): ten arguments in the CreateCommittedResource3 order: heap properties,
// heap flags (8, ALLOW_DISPLAY, for the 4K render targets), resource desc, an initial state or
// layout, clear value, then three zero words (protected session, castable count and formats in the
// desktop method), riid and ppv. Desktop: CreateCommittedResource with the sanitized desc in
// COMMON, GENERIC_READ on an upload heap and COPY_DEST on a readback heap (the initial states D3D12
// requires there; the Windows 10 runtime refuses COMMON with E_INVALIDARG). When desktop rejects the
// clear value it is retried without one.
HRESULT __stdcall device_create_committed_x(Wrapper* self, const D3D12_HEAP_PROPERTIES* props, UINT heap_flags,
                                            const D3D12_RESOURCE_DESC* desc, UINT state, const D3D12_CLEAR_VALUE* clear,
                                            void* session, UINT castable_count, const void* castable, const GUID* iid,
                                            void** out)
{
    char text[48];
    guid_text(iid, text, sizeof(text));
    if (!out || !props || !desc || !iid) return E_POINTER;
    *out = nullptr;
    const D3D12_RESOURCE_DESC copy = sanitized(*desc);
    const auto flags = static_cast<D3D12_HEAP_FLAGS>(heap_flags);
    const D3D12_RESOURCE_ALLOCATION_INFO info = device_of(self)->GetResourceAllocationInfo(0, 1, &copy);
    check_device("Device before slot", 127);
    void* const title_memory = info.SizeInBytes != ~0ull ? title_resource_memory(info.SizeInBytes, info.Alignment) : nullptr;
    D3D12_CLEAR_VALUE mapped_clear{};
    if (clear) { mapped_clear = *clear; mapped_clear.Format = desktop_format(mapped_clear.Format); clear = &mapped_clear; }
    ID3D12Resource* resource = nullptr;
    const D3D12_RESOURCE_STATES initial = props->Type == D3D12_HEAP_TYPE_UPLOAD     ? D3D12_RESOURCE_STATE_GENERIC_READ
                                          : props->Type == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST
                                                                                    : D3D12_RESOURCE_STATE_COMMON;
    D3D12_CLEAR_VALUE clear_copy{};
    const auto* clear_value = packed_clear_value(clear, desktop_format(desc->Format), copy.Format, clear_copy);
    HRESULT hr = kind_for(*iid) == resource_kind
        ? device_of(self)->CreateCommittedResource(props, flags, &copy, initial, clear_value, IID_PPV_ARGS(&resource))
        : E_NOINTERFACE;
    bool cleared = clear != nullptr;
    if (FAILED(hr) && clear && kind_for(*iid) == resource_kind) {
        hr = device_of(self)->CreateCommittedResource(props, flags, &copy, initial, nullptr, IID_PPV_ARGS(&resource));
        cleared = false;
    }
    static LONG calls;
    if (first_calls(calls, 12))
        log("x2w-d3d12 Device::slot127(CreateCommittedResource3?) heap=%d flags=0x%x dim=%d %llux%u fmt=%d rflags=0x%x "
            "state=0x%x clear=%p%s session=%p castable=%u/%p riid=%s -> 0x%08lx", props->Type, heap_flags, desc->Dimension,
            desc->Width, desc->Height, desc->Format, desc->Flags, state, static_cast<const void*>(clear),
            clear && !cleared ? "(dropped)" : "", session, castable_count, castable, text, static_cast<unsigned long>(hr));
    if (FAILED(hr)) return hr;
    x2w_startup::mark_packed_stand_in(resource, desktop_format(desc->Format));
    Wrapper* w = wrap(resource, resource_kind);
    if (copy.Format != desc->Format && copy.Format == x2w_srgb::family(desc->Format))
        w->own().srgb_uav_format = static_cast<unsigned char>(desc->Format);
    w->own().budget = x2w_budget::committed(device_of(self), resource, props->Type);
    w->own().title_memory = title_memory;
    retire_set_class(w, *props, flags, copy, title_memory);
    note_resource(w, desc, info.SizeInBytes != ~0ull ? info.SizeInBytes : 0);
    *out = w;
    return S_OK;
}

// Device slot 80 (Xbox order): (type, timeout, 0, flags, &token) -> HRESULT, the frame-event wait
// that hands out a frame token (called with type 2 / timeout -1 and type 0 / timeout 0, flags 0/2).
// Paces to the interval from slot 78 (33.3 ms by default) and returns increasing tokens.
LONG64 frame_token;
LONG frame_waits[4];
LONG64 frame_wait_ms;
HRESULT __stdcall device_wait_frame_event(Wrapper*, UINT type, UINT timeout, void* reserved, UINT flags, UINT64* token)
{
    Timed timed(time_wait);
    // One timeline: the origin event (type 0) fires once per frame interval; any other event type
    // fires at the current origin plus the offset it was scheduled with, so waiting for it after the
    // origin does not cost another whole interval.
    static LARGE_INTEGER frequency, origin;
    static SRWLOCK lock = SRWLOCK_INIT;
    AcquireSRWLockExclusive(&lock);
    if (!frequency.QuadPart) {
        QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&origin);
    }
    // LOCAL: X2W_FRAME_US overrides the title's interval (for example 8333 for 120 Hz, 1 for no pacing)
    // so boot, menus and loading run faster in test runs.
    static const UINT override_us = [] {
        char value[16]{};
        return GetEnvironmentVariableA("X2W_FRAME_US", value, sizeof(value)) ? static_cast<UINT>(std::atol(value)) : 0u;
    }();
    const UINT interval_us = override_us ? override_us : frame_interval_us ? frame_interval_us : 33333;
    const LONGLONG interval = frequency.QuadPart * interval_us / 1000000;
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    LONGLONG target;
    if (type == 0) {
        target = origin.QuadPart + interval;
        if (target < now.QuadPart) target = now.QuadPart;  // running late: start a new timeline here
        origin.QuadPart = target;
    } else {
        const UINT offset_us = frame_event_offset_us[type < 4 ? type : 3];
        target = origin.QuadPart + frequency.QuadPart * (override_us && offset_us > override_us ? override_us : offset_us) / 1000000;
    }
    const LONGLONG wait_ticks = target > now.QuadPart ? target - now.QuadPart : 0;
    ReleaseSRWLockExclusive(&lock);
    if (timeout != 0 && wait_ticks > 0) {
        // High-resolution timer where available (Sleep is too coarse for a 33 ms frame).
        X2W_ZONE("frame wait (sleep)");
        static thread_local HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, 0x00000002 /* high resolution */, TIMER_ALL_ACCESS);
        LARGE_INTEGER due;
        due.QuadPart = -(wait_ticks * 10000000 / frequency.QuadPart);
        if (timer && due.QuadPart < 0 && SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) WaitForSingleObject(timer, 1000);
        else Sleep(static_cast<DWORD>(wait_ticks * 1000 / frequency.QuadPart));
    }
    InterlockedIncrement(&frame_waits[type < 4 ? type : 3]);
    InterlockedExchangeAdd64(&frame_wait_ms, timeout != 0 ? wait_ticks * 1000 / frequency.QuadPart : 0);
    const UINT64 value = static_cast<UINT64>(InterlockedIncrement64(&frame_token));
    if (token) *token = value;
    static LONG calls;
    if (first_calls(calls, 4))
        log("x2w-d3d12 Device::slot80(frame event wait) type=%u timeout=0x%x reserved=%p flags=0x%x -> token %llu", type, timeout,
            reserved, flags, value);
    return S_OK;
}

// Diagnostic overlay: while nothing is rendered, the title window would stay blank. Every few
// presents a status text is drawn into it with GDI so the state is visible. This is not game output.
LONG draw_calls_seen;
void draw_overlay(LONG frame, double seconds)
{
    HWND window = x2w_startup::overlay_window();
    if (!window) return;
    HDC dc = GetDC(window);
    if (!dc) return;
    RECT client{};
    GetClientRect(window, &client);
    bool pictured = false;
    if (render_mode()) {
        AcquireSRWLockShared(&shown_lock);
        if (!shown_pixels.empty()) {
            BITMAPINFO info{};
            info.bmiHeader.biSize = sizeof(info.bmiHeader);
            info.bmiHeader.biWidth = shown_width;
            info.bmiHeader.biHeight = -shown_height;
            info.bmiHeader.biPlanes = 1;
            info.bmiHeader.biBitCount = 32;
            SetStretchBltMode(dc, COLORONCOLOR);
            StretchDIBits(dc, 0, 0, client.right, client.bottom, 0, 0, shown_width, shown_height, shown_pixels.data(), &info,
                          DIB_RGB_COLORS, SRCCOPY);
            pictured = true;
        }
        ReleaseSRWLockShared(&shown_lock);
    }
    HBRUSH background = CreateSolidBrush(RGB(16, 18, 24));
    if (pictured) {
        RECT box{client.left + 12, client.top + 12, client.left + 760, client.top + 190};
        FillRect(dc, &box, background);
    } else {
        FillRect(dc, &client, background);
    }
    DeleteObject(background);
    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, RGB(230, 232, 236));
    char text[640];
    std::snprintf(text, sizeof(text),
                  "Xbox2Windows startup experiment (diagnostic overlay)\n\n"
                  "Frame %ld presented, %.1f s since the first, %.1f fps; %ld draw and state calls recorded.\n"
                  "%s%s",
                  frame, seconds, seconds > 0 ? frame / seconds : 0.0, draw_calls_seen,
                  render_mode() ? "Rendering through translated pipelines; the picture behind is the presented surface: " : "",
                  render_mode() ? shown_note : "");
    RECT area{client.left + 24, client.top + 24, client.right - 24, client.bottom - 24};
    DrawTextA(dc, text, -1, &area, DT_LEFT | DT_TOP | DT_WORDBREAK);
    ReleaseDC(window, dc);
}

#include "startup_stubs/d3d12_compute.inl"
#include "startup_stubs/d3d12_heap_raw.inl"  // LOCAL: `heapraw 1`, raw V#s in the title heap
#include "startup_stubs/d3d12_range_alias.inl"  // LOCAL indirect: X2W_RANGE_ALIAS
// Translated shader ABI (X2W_GPU_ABI=1): the owners' bodies (gpu_hooks.h), then WP0's installed handlers.
#include "startup_stubs/gpu_pipeline.inl"  // WP2 vpipe
#include "startup_stubs/gpu_heap.inl"      // WP5 rawtwins
#include "startup_stubs/gpu_views.inl"     // WP5 rawtwins
#include "startup_stubs/gpu_find.h"        // WP3 vbind
#include "startup_stubs/gpu_bind.inl"      // WP3 vbind
#include "startup_stubs/gpu_indirect.inl"  // WP4 vindirect
#include "startup_stubs/gpu_hooks.inl"     // WP0 vhooks
#include "startup_stubs/d3d12_vdesc.inl"  // LOCAL meshrt: `heapwords 1`
#include "startup_stubs/d3d12_gpu_time.inl"
#include "startup_stubs/d3d12_indirect.inl"
#include "startup_stubs/d3d12_mesh.inl"  // LOCAL meshrt
#include "startup_stubs/d3d12_record_trace.inl"  // LOCAL (scene): X2W_MESH_RECORD_TRACE
#include "startup_stubs/cull_input_capture.inl"  // LOCAL (diagnostic): X2W_CULL_INPUT_CAPTURE
#include "startup_stubs/d3d12_rawaddr.inl"
#include "startup_stubs/d3d12_present.inl"
#include "startup_stubs/d3d12_copy_buffer.inl"

// LOCAL stall: every 300 presents (or 10 s), one line: the wall time between them on the presenting thread, split
// into time inside timed stand-in calls and outside, the thread's own CPU time in the same window (wall
// minus CPU is time it was blocked or not scheduled), and the threads with the most stand-in time in the
// window. Says whether the frame period is the presenting thread's own work or a wait on something else.
// X2W_PRESENT_SPLIT=<frames>: another period (default 300); 0: off.
LONG present_split_period()
{
    static const LONG period = [] {
        char value[16]{};
        return GetEnvironmentVariableA("X2W_PRESENT_SPLIT", value, sizeof(value)) ? std::atol(value) : 300L;
    }();
    return period;
}

void present_split(LONG frame)
{
    static LARGE_INTEGER frequency, last;
    static LONG last_frame;
    static LONG64 last_inside, last_ticks[thread_inside_max];
    static ULONG64 last_cpu;
    static DWORD last_tid;
    if (!frequency.QuadPart) QueryPerformanceFrequency(&frequency);
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    FILETIME created, exited, kernel, user;
    GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user);
    const auto u64_of = [](const FILETIME& t) { return (static_cast<ULONG64>(t.dwHighDateTime) << 32) | t.dwLowDateTime; };
    const ULONG64 cpu = u64_of(kernel) + u64_of(user);  // 100 ns units
    const DWORD tid = GetCurrentThreadId();
    const LONG64 inside = inside_slot()->ticks;
    const LONG count = thread_inside_count < thread_inside_max ? thread_inside_count : thread_inside_max;
    const auto ms = [](LONG64 ticks) { return static_cast<double>(ticks) * 1000.0 / static_cast<double>(frequency.QuadPart); };
    if (last.QuadPart && tid == last_tid && frame > last_frame) {
        const double wall = ms(now.QuadPart - last.QuadPart), in = ms(inside - last_inside);
        const double cpu_ms = static_cast<double>(cpu - last_cpu) / 1e4;
        struct Top { LONG tid; LONG64 ticks; } top[4]{};
        for (LONG i = 0; i < count; ++i) {
            const LONG64 delta = thread_inside[i].ticks - last_ticks[i];
            for (int j = 0; j < 4; ++j) {
                if (delta <= top[j].ticks) continue;
                for (int m = 3; m > j; --m) top[m] = top[m - 1];
                top[j] = {thread_inside[i].tid, delta};
                break;
            }
        }
        char text[160];
        int used = 0;
        text[0] = 0;
        for (const Top& t : top)
            if (t.ticks) used += std::snprintf(text + used, sizeof(text) - used, " %ld:%.0f", t.tid, ms(t.ticks));
        const LONG frames = frame - last_frame;
        log("x2w-present split over %ld frames (%.1f ms/frame): presenting thread %lu inside timed d3d12_xs calls %.1f "
            "ms/frame, outside %.1f ms/frame (xmem/XCRDAPI time counts as outside); its CPU %.1f ms/frame, so %.1f ms/frame "
            "blocked or not running; timed d3d12_xs ms by thread (tid:ms over the window):%s; %ld threads seen",
            frames, wall / frames, tid, in / frames, (wall - in) / frames,
            cpu_ms / frames, (wall - cpu_ms) / frames, text, count);
        fence_times::report();  // X2W_FENCE_TIMES=1, same window
    }
    last = now;
    last_frame = frame;
    last_inside = inside;
    last_cpu = cpu;
    last_tid = tid;
    for (LONG i = 0; i < count; ++i) last_ticks[i] = thread_inside[i].ticks;
}

// Render mode: a coarse signature of the presented picture for input-script waits (present_signature.h,
// present_signature_sampler.h). The block always counts presents; samples are taken every 4 presents
// only while a script with a picture wait has asked for them (request_sampling). X2W_PRESENT_SIGNATURE_EVERY=<n>
// samples every n presents from the start without a request, and 0 turns the signature off.
// 18 one-texel rows are copied on the presenting queue and read back on a later present; nothing waits.
void present_signature(Wrapper* queue, Wrapper* surface, LONG frame)
{
    static const int explicit_every = [] {
        char value[16]{};
        return GetEnvironmentVariableA("X2W_PRESENT_SIGNATURE_EVERY", value, sizeof(value)) ? std::atoi(value) : -1;
    }();
    if (explicit_every == 0) return;
    static const int every = explicit_every > 0 ? explicit_every : 4;
    static x2w_present_signature::Block* const block = [] {
        auto* created = x2w_present_signature::create_block(GetCurrentProcessId(), static_cast<std::uint32_t>(every));
        log("x2w-present picture signature: Local\\x2w_present_signature_%lu, every %d presents %s",
            static_cast<unsigned long>(GetCurrentProcessId()), every,
            explicit_every > 0 ? "from the start (X2W_PRESENT_SIGNATURE_EVERY)" : "once a script wait asks for it");
        return created;
    }();
    // Kept until the process ends, so a copy in flight never loses the objects it uses.
    static auto* const sampler = new x2w_present_signature::Sampler(every, explicit_every > 0, red_producers().enabled());
    ID3D12Resource* real = surface && surface->own().kind == resource_kind ? static_cast<ID3D12Resource*>(surface->own().real) : nullptr;
    const bool published = sampler->present(device_of(the_device), static_cast<ID3D12CommandQueue*>(queue->own().real), real, frame, block);
    static LONG first, refused, failed;
    if (published && red_producers().enabled() && sampler->last_format() == DXGI_FORMAT_R8G8B8A8_UNORM) {
        const auto* rgba = sampler->last_rgba();
        const bool red = rgba[0] == 255 && rgba[1] == 0 && rgba[2] == 0;
        const bool matched = red_producers().matches(rgba);
        red_producers().observe(sampler->last_present(), frame, sampler->last_surface(), matched);
        static bool was_matched;
        if (matched != was_matched || red_producers().interested()) {
            log("x2w-red sample present=%lld published_at=%ld target=%p rgba=%u,%u,%u,%u red=%u match=%u tick=%llu",
                sampler->last_present(), frame, sampler->last_surface(), rgba[0], rgba[1], rgba[2], rgba[3], red, matched, GetTickCount64());
            was_matched = matched;
        }
    }
    if (published && first_calls(first, 1))
        log("x2w-present picture signature: first sample published at present %ld in Local\\x2w_present_signature_%lu (every %d presents)",
            frame, static_cast<unsigned long>(GetCurrentProcessId()), every);
    if (sampler->unsupported_format() != DXGI_FORMAT_UNKNOWN && first_calls(refused, 1))
        log("x2w-present picture signature: surface format %d is not sampled", sampler->unsupported_format());
    if (sampler->failed() && first_calls(failed, 1))
        log(sampler->stranded() ? "x2w-present picture signature: Signal failed after a copy was submitted; sampling stopped, its surface stays referenced"
                                : "x2w-present picture signature: setup failed, no samples");
}

void report_budget(ID3D12CommandQueue* queue)
{
    using Microsoft::WRL::ComPtr;
    ComPtr<ID3D12Device> device;
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter3> adapter;
    if (SUCCEEDED(queue->GetDevice(IID_PPV_ARGS(&device))) && SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        const LUID luid = device->GetAdapterLuid();
        if (FAILED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter)))) {
            ComPtr<IDXGIAdapter1> warp;
            DXGI_ADAPTER_DESC1 desc{};
            if (SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp))) && SUCCEEDED(warp->GetDesc1(&desc)) &&
                desc.AdapterLuid.LowPart == luid.LowPart && desc.AdapterLuid.HighPart == luid.HighPart)
                warp.As(&adapter);
        }
    }
    char local[48] = "n/a", nonlocal[48] = "n/a";
    const auto sample = [&](DXGI_MEMORY_SEGMENT_GROUP segment, char (&text)[48]) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info{};
        if (adapter && SUCCEEDED(adapter->QueryVideoMemoryInfo(0, segment, &info)))
            std::snprintf(text, sizeof(text), "%llu/%llu", info.CurrentUsage, info.Budget);
    };
    sample(DXGI_MEMORY_SEGMENT_GROUP_LOCAL, local);
    sample(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, nonlocal);
    const auto counts = x2w_budget::snapshot();
    char text[640]{};
    std::size_t used = 0;
    for (int i = 0; i < x2w_budget::category_count; ++i) {
        const auto& active = counts.live[i];
        const auto& held = counts.retained[i];
        const int n = std::snprintf(text + used, sizeof(text) - used, " %s=%llu/%llu+%llu/%llu", x2w_budget::names[i],
                                    active.bytes, active.objects, held.bytes, held.objects);
        if (n < 0 || static_cast<std::size_t>(n) >= sizeof(text) - used) break;
        used += n;
    }
    log("x2w-budget local=%s nonlocal=%s%s unmeasured=%llu", local, nonlocal, text, counts.unmeasured);
}

// Queue slot 32 (Xbox order): (plane count, plane parameters*, present parameters*) -> HRESULT, the
// console present. Nothing is shown yet; frames are counted and a heartbeat is logged.
HRESULT __stdcall queue_present(Wrapper* self, UINT planes, const void* plane_parameters, const void* present_parameters)
{
    // First, before anything reads or copies the surface: a presented resource keeps its old lifetime.
    if (release_retired_enabled() && planes && plane_parameters) {
        const auto* words = static_cast<const u64*>(plane_parameters);
        if (x2w_local::readable(words, 0x18) && words[1] >= 1 && x2w_local::readable(reinterpret_cast<const void*>(words[2]), 8))
            retire_leave_class(*reinterpret_cast<void* const*>(words[2]), RetireLeft::present);
    }
    drain_retired_resources();
    Timed timed(time_present);
    X2W_ZONE("Present");
    X2W_FRAME_MARK;
    static LONG frames;
    static ULONGLONG started, last_report;
    const LONG frame = InterlockedIncrement(&frames);
    if (const LONG period = present_split_period(); period > 0) {
        static ULONGLONG split_due;
        const ULONGLONG tick = GetTickCount64();
        if (frame == 1 || frame % period == 0 || tick >= split_due) {
            split_due = tick + 10000;
            present_split(frame);
        }
    }
    frame_number = frame;
    const bool gpu_trace_end = gpu_trace_frame == frame;
    if (vbind::compute_capture_in_flight.load() && TryAcquireSRWLockExclusive(&draw_lock)) {
        vbind::compute_capture_poll_locked();
        ReleaseSRWLockExclusive(&draw_lock);
    }
    if (range_watch_count()) range_watch_sample("present");
    note_present_planes(frame, planes, plane_parameters, present_parameters);
    note_quiet_frame(frame);
    note_descriptor_watch(frame);
    draw_dump_flush();
    red_producers().flush();
    cull_capture_flush();  // LOCAL: X2W_CULL_INPUT_CAPTURE
    cx_dump::poll();  // LOCAL lt-lead probe: X2W_CX_DUMP_INPUTS read-back
    dd_dump::poll();  // LOCAL lt-lead probe: X2W_DRAW_DUMP_INPUTS read-back
    {
        auto& scene = gpu_scene_trace();
        if (scene.mrt || !scene.request_file.empty()) {
            LONG completed = -1;
            {
                std::lock_guard<std::mutex> held(scene.lock);
                if (scene.active_frame && scene.active_frame <= frame) {
                    completed = scene.active_id;
                    InterlockedExchange(&gpu_trace_request_id, 0);
                    scene.active_frame = scene.active_id = 0;
                    std::atomic_store(&gpu_scene_stage_override, TraceStageKey{});
                    std::atomic_store(&gpu_scene_second_override, TraceStageKey{});
                    InterlockedExchange(&gpu_trace_frame, 0);
                    InterlockedExchange(&trace_frame_active, 0);
                    InterlockedExchange(&trace_uav_busy_frame, 0);
                }
            }
            if (completed >= 0) {
                if (x2w_local::env_on("X2W_D3D12_TRACE_TARGETS"))
                    summarise_traced_targets(static_cast<ID3D12CommandQueue*>(self->own().real), completed);
                log("x2w-frame ---- end deferred frame %ld request %ld", frame, completed);
            }
            scene.poll();
        }
        const LONG wanted = trace_requested_frame();
        // The call that presents frame n-1 starts the trace; the one that presents frame n ends it.
        if (wanted && !gpu_trace_frame && frame == wanted - 1) {
            log("x2w-frame ---- begin frame %ld", wanted);
            reset_gpu_trace_commands();
            InterlockedExchange(&gpu_trace_frame, wanted);
            InterlockedExchange(&trace_frame_active, 1);
        } else if (wanted && gpu_trace_frame == wanted && frame == wanted) {
            InterlockedExchange(&trace_frame_active, 0);
            InterlockedExchange(&gpu_trace_frame, 0);
            if (x2w_gpu::gpu_abi_on() && x2w_local::env_on("X2W_D3D12_TRACE_TARGETS"))
                summarise_traced_targets(static_cast<ID3D12CommandQueue*>(self->own().real));
            log("x2w-frame ---- end frame %ld", wanted);
        }
        // LOCAL: X2W_D3D12_TRACE_BUSY=n traces one frame after the first frame with at least n dispatches
        // (the first busy frame after loading), then one more every 3000 frames.
        static const LONG busy = [] {
            char value[16]{};
            return GetEnvironmentVariableA("X2W_D3D12_TRACE_BUSY", value, sizeof(value)) ? std::atol(value) : 0L;
        }();
        static LONG busy_end, busy_next;
        const LONG dispatched = InterlockedExchange(&frame_dispatches, 0);
        mesh_stats_present(frame);  // LOCAL meshrt: traced mesh calls' pipeline statistics, 8 presents after the traced frame
        mesh_fs_present();          // LOCAL meshrt: X2W_MESH_STATS_FRAMES
        if (busy && busy_end && frame == busy_end) {
            InterlockedExchange(&gpu_trace_frame, 0);
            InterlockedExchange(&trace_uav_busy_frame, 0);
            InterlockedExchange(&trace_frame_active, 0);
            log("x2w-frame ---- end busy frame %ld", frame);
            // The target read-back stalls Present for about a second: only with X2W_D3D12_TRACE_TARGETS=1.
            static const bool targets = x2w_local::env_on("X2W_D3D12_TRACE_TARGETS");
            record_trace_report(static_cast<ID3D12CommandQueue*>(self->own().real));  // LOCAL (scene): X2W_MESH_RECORD_TRACE
            if (targets) summarise_traced_targets(static_cast<ID3D12CommandQueue*>(self->own().real));
            summarise_indirect_calls();
            busy_end = 0;
        } else if (busy && !scene.mrt && !scene.waiting.load() && !gpu_trace_frame && !busy_end && frame >= busy_next) {
            // LOCAL (scene): X2W_D3D12_TRACE_BUSY_AFTER=n delays the traced frame by n frames after the first busy
            // frame (the scene's first frames are black while its pipelines still compile).
            static const LONG after = [] {
                char value[16]{};
                return GetEnvironmentVariableA("X2W_D3D12_TRACE_BUSY_AFTER", value, sizeof(value)) ? std::atol(value) : 0L;
            }();
            static LONG busy_seen;
            if (!busy_seen && dispatched >= busy) busy_seen = frame;
            if (busy_seen && frame >= busy_seen + after) {
                log("x2w-frame ---- begin busy frame %ld (previous frame: %ld dispatches; first busy frame %ld)", frame + 1, dispatched, busy_seen);
                reset_gpu_trace_commands();
                InterlockedExchange(&gpu_trace_frame, frame + 1);
                InterlockedExchange(&trace_frame_active, 1);
                busy_end = frame + 1;
                InterlockedExchange(&trace_uav_busy_frame, busy_end);
                busy_next = frame + 3000;
                busy_seen = 0;
            }
        }
    }
    bool shown_in_window = false;
    {
        Wrapper* surface = nullptr;
        const auto* words = static_cast<const u64*>(plane_parameters);
        if (planes && words && x2w_local::readable(words, 0x18) && words[1] >= 1 && x2w_local::readable(reinterpret_cast<const void*>(words[2]), 8))
            surface = as_wrapper(*reinterpret_cast<void* const*>(words[2]));
        capture::present(frame, surface);
        if (gpu_trace_end && surface && surface->own().kind == resource_kind)
            log("x2w-frame gpu present frame %ld: title=%p desktop=%p", frame, static_cast<void*>(surface),
                static_cast<void*>(surface->own().real));
        if (surface && surface->own().kind == resource_kind)
            red_producers().present(static_cast<ID3D12Resource*>(surface->own().real));
        // Render mode: the presented surface goes to the title's window through a swap chain.
        if (render_mode() && surface && surface->own().kind == resource_kind && surface->own().real)
            shown_in_window = present_to_window(static_cast<ID3D12CommandQueue*>(self->own().real), static_cast<ID3D12Resource*>(surface->own().real));
        InterlockedExchange(&last_frame_shown, shown_in_window ? 1 : 0);
        if (render_mode()) present_signature(self, surface, frame);
    }
    const ULONGLONG now = GetTickCount64();
    if (frame == 1) started = last_report = now;
    if (frame <= 2) {
        char a[3 * 0x50 + 1], b[3 * 0x40 + 1];
        hex_text(plane_parameters, 0x50, a, sizeof(a));
        hex_text(present_parameters, 0x40, b, sizeof(b));
        log("x2w-d3d12 CommandQueue::slot32(present) planes=%u plane=[%s] present=[%s]", planes, a, b);
    }
    // LOCAL: X2W_D3D12_SHOT_FRAMES=<n> reads the presented surface back (and writes a numbered shot) every
    // n frames while the window shows frames, instead of every 60 (0.5 s at 120 fps): the setup driver
    // waits for fresh shots between presses.
    static const ULONGLONG shot_frames = [] {
        char value[8]{};
        const DWORD n = GetEnvironmentVariableA("X2W_D3D12_SHOT_FRAMES", value, sizeof(value));
        const unsigned long frames = n && n < sizeof(value) ? std::strtoul(value, nullptr, 10) : 0ul;
        return frames ? static_cast<ULONGLONG>(frames) : 60ull;
    }();
    if (render_mode()) read_back_presented(nullptr, nullptr, false);  // LOCAL stall: picks up a finished copy
    const bool paired = red_producers().pair_present(frame, static_cast<ID3D12CommandQueue*>(self->own().real));
    // Wall-clock sampling also captures transitions that present only a few frames per minute.
    static const DWORD shot_ms = [] {
        char value[12]{};
        const DWORD n = GetEnvironmentVariableA("X2W_D3D12_SHOT_MS", value, sizeof(value));
        return n && n < sizeof(value) ? std::strtoul(value, nullptr, 10) : 0ul;
    }();
    static std::atomic<ULONGLONG> last_shot{0};
    auto previous = last_shot.load(std::memory_order_relaxed);
    const bool due = shot_ms ? now >= previous && now - previous >= shot_ms &&
        last_shot.compare_exchange_strong(previous, now, std::memory_order_relaxed)
        : frame % (shown_in_window ? shot_frames : 6) == 0;
    if (render_mode() && (paired || due) && planes && plane_parameters) {
        // Plane parameters: +0x00 frame token, +0x08 resource count, +0x10 pointer to the resources.
        const auto* words = static_cast<const u64*>(plane_parameters);
        if (x2w_local::readable(words, 0x18) && words[1] >= 1 && x2w_local::readable(reinterpret_cast<const void*>(words[2]), 8)) {
            Wrapper* surface = as_wrapper(*reinterpret_cast<void* const*>(words[2]));
            if (surface && surface->own().kind == resource_kind && surface->own().real)
                read_back_presented(static_cast<ID3D12CommandQueue*>(self->own().real), static_cast<ID3D12Resource*>(surface->own().real), true, paired ? frame : 0);
            else
                std::snprintf(shown_note, sizeof(shown_note), "presented object %p is not a known resource",
                              *reinterpret_cast<void* const*>(words[2]));
        }
    }
    if (shown_in_window) {
        if (!present_in_title_window() && frame % 30 == 0) {
            // About 100 fixed characters, the revision (at most 46) and three numbers fit in 256.
            wchar_t text[256];
            swprintf(text, 256, L"Xbox2Windows %ls - frame %ld, %.1f fps, %ld translated draws (title bar is diagnostics, picture is game output)",
                     x2w_build_rev_w, frame, frame * 1000.0 / (now - started ? now - started : 1), translated_draws);
            SetWindowTextW(output_window(), text);
        }
    } else if (frame % 10 == 1 || (render_mode() && frame % 6 == 0)) {
        draw_overlay(frame, (now - started) / 1000.0);
    }
    // The null-binding total and its causes change together under draw_lock (draw_translated holds it), so they are copied there.
    LONG null_total = 0, null_causes[x2w_local::null_cause_count] = {};
    if (render_mode() && frame % 300 == 0) {
        lock_draw(0 + __LINE__);
        null_total = null_bindings;
        for (int c = 0; c < x2w_local::null_cause_count; ++c) null_causes[c] = null_by_cause[c];
        ReleaseSRWLockExclusive(&draw_lock);
    }
    if (render_mode() && frame % 300 == 0)
        log("x2w-render totals: resets %ld (failed %ld), closes %ld, clears %ld, target binds %ld (unresolved %ld), executes %ld, draws %ld (skipped %ld), "
            "translated draws %ld (not drawn %ld), pipelines translated %ld / missing %ld, null bindings %ld (entry %ld, buffer %ld, token %ld, wrapper %ld, dimension %ld), "
            "buffer links %ld, refused-view entries %ld (nulled %ld, rewritten markers %ld), textures uploaded %ld (%ld KiB, unreadable %ld), heap-indexed roots deferred %ld (real draws refused: view heap %ld, "
            "sampler heap %ld, both heaps %ld, state %ld), pipelines first looked up by ExecuteIndirect %ld, compute requests published %ld (already present %ld, failed %ld, pointer-like blob qwords %ld)", render_resets,
            render_reset_failures, render_closes, render_clears, render_target_binds, render_target_unresolved, render_executes, render_draws, render_draws_skipped,
            translated_draws, untranslated_draws, translation_ready, translation_missing, null_total,
            null_causes[0], null_causes[1], null_causes[2], null_causes[3], null_causes[4], buffer_links, buffer_refused_views, buffer_links_nulled, marker_rewrites, textures_uploaded,
            texture_bytes_uploaded, textures_unreadable, heap_roots_deferred,
            heap_root_refusals[static_cast<int>(x2w_startup::RealRootDecision::view_heap_missing)],
            heap_root_refusals[static_cast<int>(x2w_startup::RealRootDecision::sampler_heap_missing)],
            heap_root_refusals[static_cast<int>(x2w_startup::RealRootDecision::both_heaps_missing)],
            heap_root_refusals[static_cast<int>(x2w_startup::RealRootDecision::state_not_restorable)], indirect_first_lookups,
            compute_requests_published, compute_requests_present, compute_requests_failed, compute_blob_pointer_words);
    if (render_mode() && frame % 300 == 0) report_target_binds();
    if (render_mode() && frame % 300 == 0) report_draw_state();  // LOCAL (renderstate): d3d12_draw.inl
    if (frame % 300 == 0) draw_lock_times::report("300 frames");  // LOCAL stall: X2W_DRAW_LOCK_TIMES
    if (x2w_local::probe_cache_on()) {  // LOCAL stall: X2W_PROBE_CACHE (local_log.h); entries also end at each present
        x2w_local::invalidate_probe_cache();
        if (frame % 300 == 0) {
            x2w_local::ProbeCache& c = x2w_local::probe_cache();
            LARGE_INTEGER f;
            QueryPerformanceFrequency(&f);
            log("x2w-probe cache (X2W_PROBE_CACHE, %llu ms, cleared at each present): since start %lld probes answered from the cache without a "
                "fault, %lld faults (%lld VirtualQuery, %.1f ms), %lld unreadable ranges kept, %lld query-first rejections", x2w_local::probe_cache_ms(), c.hits, c.faults,
                c.queries, static_cast<double>(c.query_ticks) * 1000.0 / static_cast<double>(f.QuadPart), c.cached, c.preflight_rejects);
        }
    }
    if (render_mode() && frame % 300 == 0) {
        // Slot-61 calls since the start, by outcome, then by signature kind (zero kinds left out).
        static const char* const outcomes[] = {"forwarded", "refused graphics-pipeline", "compute-pipeline",
                                               "placeholder-signature", "other"};
        static const char* const kinds[] = {"other", "dispatch", "draw", "drawindexed", "dispatchmesh"};
        char line[1024];  // fits every outcome with five 19-digit counts
        int n = std::snprintf(line, sizeof(line), "x2w-render indirect:");
        for (int o = 0; o < indirect_outcomes; ++o) {
            n += std::snprintf(line + n, sizeof(line) - n, "%s %s:", o ? ";" : "", outcomes[o]);
            const int start = n;
            for (int k = 0; k < 5; ++k)
                if (const LONG64 calls = indirect_calls[o][k])
                    n += std::snprintf(line + n, sizeof(line) - n, " %s %lld", kinds[k], static_cast<long long>(calls));
            if (n == start) n += std::snprintf(line + n, sizeof(line) - n, " 0");
        }
        log("%s", line);
    }
    if (render_mode() && frame % 300 == 0) {
        // Which command-list slots the title called in the last 300 frames, per frame.
        char line[2048];
        int n = std::snprintf(line, sizeof(line), "x2w-render list slots per frame:");
        for (int i = 0; i < 256 && n < static_cast<int>(sizeof(line)) - 24; ++i) {
            const LONG64 calls = InterlockedExchange64(&list_slot_census[i], 0);
            if (calls) n += std::snprintf(line + n, sizeof(line) - n, " %d:%.1f", i, calls / 300.0);
        }
        log("%s", line);
        report_dispatches();
        report_indirect(300);  // LOCAL: d3d12_indirect.inl
        report_mesh(300);  // LOCAL meshrt: d3d12_mesh.inl (X2W_MESH=1)
        regblob::report();
        report_cx_views();
        gpu_time::report();  // LOCAL: X2W_CX_GPU_TIME
    }
    if (now - last_report >= 5000) {
        last_report = now;
        log("x2w-d3d12 views noted %lld (distinct %lld), descriptors copied %lld, heaps %llu, placed buffers %lld "
            "(region-information proofs %lld, region queries %lld)", views_noted, views_distinct, descriptors_copied,
            static_cast<u64>(heap_infos().size()), static_cast<long long>(placed_buffer_calls),
            static_cast<long long>(placed_buffer_region_info), static_cast<long long>(placed_buffer_queries));
        {
            LARGE_INTEGER frequency;
            QueryPerformanceFrequency(&frequency);
            char text[768];
            int used = 0;
            for (int i = 0; i < time_count; ++i) {
                const LONG64 ticks = InterlockedExchange64(&time_ticks[i], 0), calls = InterlockedExchange64(&time_calls[i], 0);
                if (calls)
                    used += std::snprintf(text + used, sizeof(text) - used, " %s %lld ms/%lld", time_names[i],
                                          static_cast<long long>(ticks * 1000 / frequency.QuadPart), static_cast<long long>(calls));
            }
            log("x2w-d3d12 time inside the stand-in since the last report (ms/calls):%s", text);
            static LONG64 placed_reported = 0;  // LOCAL: slot 133 totals, when they changed
            const LONG64 placed_calls = placed_calls_total;
            if (placed_calls != placed_reported) {
                placed_reported = placed_calls;
                log("x2w-d3d12 placed resources (slot 133) since the start: %lld calls, %lld ms inside", static_cast<long long>(placed_calls),
                    static_cast<long long>(placed_ticks_total * 1000 / frequency.QuadPart));
            }
            used = 0;
            text[0] = 0;
            for (int i = 0; i < 256; ++i) {
                const LONG64 ticks = InterlockedExchange64(&list_slot_ticks[i], 0), calls = InterlockedExchange64(&list_slot_calls[i], 0);
                if (calls && ticks * 1000 / frequency.QuadPart >= 20 && used < 440)
                    used += std::snprintf(text + used, sizeof(text) - used, " slot%d %lld ms/%lld", i,
                                          static_cast<long long>(ticks * 1000 / frequency.QuadPart), static_cast<long long>(calls));
            }
            if (used) log("x2w-d3d12 command-list slots over 20 ms:%s", text);
        }
        rawaddr_report();  // LOCAL raw-address shaders (d3d12_rawaddr.inl)
        report_srv_copies();  // LOCAL stall: X2W_DRAW_SRV_STATS / X2W_DRAW_SRV_IN_PLACE (d3d12_draw.inl)
        report_cx_async();    // LOCAL stall: X2W_CX_ASYNC (d3d12_compute.inl)
        report_heap_raw();    // LOCAL: `heapraw 1` drains and evictions (d3d12_heap_raw.inl)
        report_prewarm_known("at the heartbeat");  // LOCAL stall: X2W_CX_PREWARM_ALL
        report_draw_async();  // LOCAL stall: X2W_DRAW_ASYNC (d3d12_draw.inl)
        memory_srv_census.report([](const char* line) { log("%s", line); }, true);
        token_null_census.report([](const char* line) { log("%s", line); }, true);
        memory_srv_writes.report([](const char* line) { log("%s", line); }, true);
        indirect_draw_refusal_census.report([](const char* line) { log("%s", line); }, true);
        indirect_other_refusal_census.report([](const char* line) { log("%s", line); }, true);
        log("x2w-d3d12 heartbeat: %ld frames presented in %.1f s (%.1f fps overall); frame waits by type %ld/%ld/%ld/%ld, "
            "%lld ms slept", frame, (now - started) / 1000.0, frame * 1000.0 / (now - started ? now - started : 1),
            frame_waits[0], frame_waits[1], frame_waits[2], frame_waits[3], static_cast<long long>(frame_wait_ms));
        if (x2w_budget::enabled()) report_budget(static_cast<ID3D12CommandQueue*>(self->own().real));
        if (retire_reasons_enabled()) {
            char line[1536];
            if (retire_reasons_line(line, sizeof(line))) log("%s", line);
        }
    }
    return S_OK;
}

// Fixtures: the census lines of the five-second report, now and whether or not they changed, one per
// line. Returns the length of the whole text and copies up to capacity - 1 bytes of it, terminated.
extern "C" __declspec(dllexport) std::size_t x2w_d3d12_census_report(char* out, std::size_t capacity)
{
    std::size_t length = 0;
    const auto emit = [&](const char* line) {
        const std::size_t bytes = std::strlen(line);
        for (std::size_t i = 0; i <= bytes; ++i, ++length)
            if (out && length + 1 < capacity) out[length] = i < bytes ? line[i] : '\n';
    };
    memory_srv_census.report(emit, false);
    token_null_census.report(emit, false);
    memory_srv_writes.report(emit, false);
    if (indirect_draw_refusal_census.total()) indirect_draw_refusal_census.report(emit, false);
    if (indirect_other_refusal_census.total()) indirect_other_refusal_census.report(emit, false);
    if (out && capacity) out[length < capacity ? length : capacity - 1] = 0;
    return length;
}

template <int K, int N>
void enable_forward() { tables[K][N] = reinterpret_cast<void*>(&forward_slot<K, N>); }
template <int K, int N, int R, int Desc = -1>
void enable_create() { tables[K][N] = reinterpret_cast<void*>(&create_slot<K, N, R, Desc>); }
template <int K>
void enable_object()
{
    // ID3D12Object: GetPrivateData, SetPrivateData, SetPrivateDataInterface, SetName.
    enable_forward<K, 3>();
    enable_forward<K, 4>();
    enable_forward<K, 5>();
    enable_forward<K, 6>();
    if (K != device_kind) tables[K][7] = reinterpret_cast<void*>(&child_get_device);
}
template <std::size_t... K>
void enable_objects(std::index_sequence<K...>) { (enable_object<static_cast<int>(K)>(), ...); }

void build_tables()
{
    fill_all_traps(std::make_index_sequence<kind_count>{});
    for (auto& table : tables) {
        table[0] = reinterpret_cast<void*>(&query_interface);
        table[1] = reinterpret_cast<void*>(&add_ref);
        table[2] = reinterpret_cast<void*>(&release);
    }
    enable_objects(std::make_index_sequence<kind_count>{});

    // Device, desktop ID3D12Device order (slots 7-47).
    enable_forward<device_kind, 7>();                    // GetNodeCount
    tables[device_kind][8] = reinterpret_cast<void*>(&device_create_command_queue);
    enable_create<device_kind, 9, 1>();                  // CreateCommandAllocator(type, riid, ppv)
    tables[device_kind][10] = reinterpret_cast<void*>(&device_create_pipeline<10>);
    tables[device_kind][11] = reinterpret_cast<void*>(&device_create_pipeline<11>);
    enable_create<device_kind, 12, 4>();                 // CreateCommandList(node, type, allocator, pso, riid, ppv)
    enable_forward<device_kind, 13>();                   // CheckFeatureSupport
    tables[device_kind][14] = reinterpret_cast<void*>(&device_create_descriptor_heap);
    tables[device_kind][15] = reinterpret_cast<void*>(&device_descriptor_increment_size);  // GetDescriptorHandleIncrementSize(type)
    enable_create<device_kind, 16, 3>();                 // CreateRootSignature(node, blob, size, riid, ppv)
    tables[device_kind][17] = reinterpret_cast<void*>(&view_slot<17, 1>);  // CreateConstantBufferView(desc, dest)
    tables[device_kind][18] = reinterpret_cast<void*>(&view_slot<18, 2>);  // CreateShaderResourceView(res, desc, dest)
    tables[device_kind][19] = reinterpret_cast<void*>(&view_slot<19, 3>);  // CreateUnorderedAccessView(res, counter, desc, dest)
    tables[device_kind][20] = reinterpret_cast<void*>(&view_slot<20, 2>);  // CreateRenderTargetView(res, desc, dest)
    tables[device_kind][21] = reinterpret_cast<void*>(&view_slot<21, 2>);  // CreateDepthStencilView(res, desc, dest)
    tables[device_kind][22] = reinterpret_cast<void*>(&view_slot<22, 1>);  // CreateSampler(desc, dest)
    tables[device_kind][23] = reinterpret_cast<void*>(&device_copy_descriptors);
    tables[device_kind][24] = reinterpret_cast<void*>(&device_copy_descriptors_simple);
    tables[device_kind][25] = reinterpret_cast<void*>(&device_resource_allocation_info);
    tables[device_kind][26] = reinterpret_cast<void*>(&device_custom_heap_properties);
    enable_create<device_kind, 27, 5, 2>();                 // CreateCommittedResource(props, flags, desc, state, clear, riid, ppv)
    enable_create<device_kind, 28, 1>();                 // CreateHeap(desc, riid, ppv)
    enable_create<device_kind, 29, 5, 2>();                 // CreatePlacedResource(heap, offset, desc, state, clear, riid, ppv)
    enable_create<device_kind, 30, 3, 0>();                 // CreateReservedResource(desc, state, clear, riid, ppv)
    tables[device_kind][34] = reinterpret_cast<void*>(&device_residency<34>);
    tables[device_kind][35] = reinterpret_cast<void*>(&device_residency<35>);
    enable_create<device_kind, 36, 2>();                 // CreateFence(initial, flags, riid, ppv)
    enable_forward<device_kind, 37>();                   // GetDeviceRemovedReason
    tables[device_kind][38] = reinterpret_cast<void*>(&device_copyable_footprints);
    tables[device_kind][39] = reinterpret_cast<void*>(&device_create_query_heap);  // CreateQueryHeap(desc, riid, ppv)
    enable_forward<device_kind, 40>();                   // SetStablePowerState
    tables[device_kind][41] = reinterpret_cast<void*>(&device_create_command_signature);                 // CreateCommandSignature(desc, root, riid, ppv)
    enable_forward<device_kind, 42>();                   // GetResourceTiling
    // Slot 43 (desktop GetAdapterLuid) is not installed: no call site has confirmed it.
    (void)&device_adapter_luid;
    // Xbox order from here.
    tables[device_kind][48] = reinterpret_cast<void*>(&device_create_command_queue_x);
    tables[device_kind][70] = reinterpret_cast<void*>(&device_memory_srv);
    if (memory_srv_buffer_on()) tables[device_kind][71] = reinterpret_cast<void*>(&device_memory_uav);  // LOCAL (video)
    if (texture_census_on()) tables[device_kind][61] = reinterpret_cast<void*>(&device_slot61_census);  // LOCAL (textures)
    tables[device_kind][73] = reinterpret_cast<void*>(&device_create_command_list_x);
    tables[device_kind][78] = reinterpret_cast<void*>(&device_frame_interval);
    tables[device_kind][79] = reinterpret_cast<void*>(&device_frame_event);
    tables[device_kind][80] = reinterpret_cast<void*>(&device_wait_frame_event);
    tables[device_kind][55] = reinterpret_cast<void*>(&device_create_derived_pipeline);
    tables[device_kind][57] = reinterpret_cast<void*>(&device_create_pipeline_placeholder<57>);
    tables[device_kind][67] = reinterpret_cast<void*>(&device_create_pipeline_placeholder<67>);
    tables[device_kind][62] = reinterpret_cast<void*>(&device_create_compute_pipeline_x);
    tables[device_kind][86] = reinterpret_cast<void*>(&device_create_pipeline_stream_x);
    for (int slot = 3; slot <= 6; ++slot) tables[pipeline_kind][slot] = reinterpret_cast<void*>(&placeholder_ok);
    for (int slot = 3; slot <= 6; ++slot) tables[command_signature_kind][slot] = reinterpret_cast<void*>(&placeholder_ok);
    tables[device_kind][108] = reinterpret_cast<void*>(&device_resource_allocation_info1);
    tables[device_kind][127] = reinterpret_cast<void*>(&device_create_committed_x);
    tables[device_kind][133] = reinterpret_cast<void*>(&device_place_resource_x);
    if (device_noop()) fill_device_noops(tables[device_kind], std::make_index_sequence<slot_count>{});
    if (list_noop()) {
        fill_noops<list_kind>(tables[list_kind], std::make_index_sequence<slot_count>{});
        tables[list_kind][7] = reinterpret_cast<void*>(&child_get_device);
    }
    list_typed[148] = reinterpret_cast<ListTyped>(&list_allocate_upload);
    if (render_mode()) {
        list_typed[9] = reinterpret_cast<ListTyped>(&list_close);
        list_typed[51] = reinterpret_cast<ListTyped>(&list_clear_depth_stencil);
        // Slot 51 is also entered through the generic invoke() path (the fixture calls list vtable[51]
        // directly), so it must be present in tables[list_kind] as well as list_typed; siblings 52-55
        // are only reached through list_typed and do not set both.
        tables[list_kind][51] = reinterpret_cast<void*>(&list_clear_depth_stencil);
        list_typed[52] = reinterpret_cast<ListTyped>(&list_clear_target);
        list_typed[15] = reinterpret_cast<ListTyped>(&list_copy_buffer_region);
        list_typed[17] = reinterpret_cast<ListTyped>(&list_copy_resource);
        list_typed[19] = reinterpret_cast<ListTyped>(&list_resolve_subresource);
        list_typed[53] = reinterpret_cast<ListTyped>(&list_clear_uav<false>);
        list_typed[54] = reinterpret_cast<ListTyped>(&list_clear_uav<true>);
        list_typed[55] = reinterpret_cast<ListTyped>(&list_discard);
        list_typed[139] = reinterpret_cast<ListTyped>(&list_reset);
        // Upstream per-recording upload pages (list_upload_pages.h) replace the local slot 148 arena only with
        // X2W_LIST_UPLOAD_PAGES=1; without it list_upload_gpu finds no pages and its users fall through.
        if (x2w_local::env_on("X2W_LIST_UPLOAD_PAGES")) list_typed[148] = reinterpret_cast<ListTyped>(&list_allocate_upload_pages);
        list_typed[12] = reinterpret_cast<ListTyped>(&list_draw);
        list_typed[14] = reinterpret_cast<ListTyped>(&list_dispatch);
        list_typed[61] = reinterpret_cast<ListTyped>(&list_execute_indirect);
        list_typed[33] = reinterpret_cast<ListTyped>(&list_compute_root);
        list_typed[35] = reinterpret_cast<ListTyped>(&list_compute_table);
        list_typed[37] = reinterpret_cast<ListTyped>(&list_compute_constant);
        list_typed[39] = reinterpret_cast<ListTyped>(&list_compute_constants);
        list_typed[41] = reinterpret_cast<ListTyped>(&list_compute_cbv);
        list_typed[43] = reinterpret_cast<ListTyped>(&list_compute_srv);
        list_typed[45] = reinterpret_cast<ListTyped>(&list_compute_uav);
        list_typed[16] = reinterpret_cast<ListTyped>(&list_copy_texture);
        list_typed[85] = reinterpret_cast<ListTyped>(&list_copy_texture_x);
        list_typed[20] = reinterpret_cast<ListTyped>(&list_topology);
        list_typed[21] = reinterpret_cast<ListTyped>(&list_viewports);
        list_typed[22] = reinterpret_cast<ListTyped>(&list_scissors);
        list_typed[23] = reinterpret_cast<ListTyped>(&list_blend_factor);
        // LOCAL merge: upstream installs slot 24 always; here with the depth group, X2W_DEPTH_STATE, or
        // X2W_UPSTREAM_STENCIL_REF=1 (slot 24 alone, upstream's OMSetStencilRef forwarding).
        if (x2w_pipeline_state::switches_from_env().depth || x2w_local::env_on("X2W_DEPTH_STATE") ||
            x2w_local::env_on("X2W_UPSTREAM_STENCIL_REF")) list_typed[24] = reinterpret_cast<ListTyped>(&list_stencil_ref);  // LOCAL (renderstate)
        if (x2w_pipeline_state::switches_from_env().depth_clear)  // LOCAL (renderstate): not through noop_slot (float argument)
            tables[list_kind][51] = reinterpret_cast<void*>(&list_clear_depth);
        list_typed[25] = reinterpret_cast<ListTyped>(&list_pipeline);
        list_typed[x2w_list_slots::resource_barrier_slot] = reinterpret_cast<ListTyped>(&list_resource_barriers);
        list_typed[32] = reinterpret_cast<ListTyped>(&list_set_descriptor_heaps);
        // Upstream desktop compute binding (slot 33 root, slot 43 upload-page SRV; feeds its ExecuteIndirect dispatch
        // forwarding) replaces the local compute root handlers only with X2W_UPSTREAM_COMPUTE_ROOT=1.
        static const bool upstream_compute_root = x2w_local::env_on("X2W_UPSTREAM_COMPUTE_ROOT");
        if (upstream_compute_root) list_typed[33] = reinterpret_cast<ListTyped>(&list_compute_root_desktop);
        list_typed[34] = reinterpret_cast<ListTyped>(&list_graphics_root);
        if (upstream_compute_root) list_typed[43] = reinterpret_cast<ListTyped>(&list_compute_upload_srv);
        list_typed[44] = reinterpret_cast<ListTyped>(&list_root_srv);
        list_typed[46] = reinterpret_cast<ListTyped>(&list_root_uav);  // LOCAL draw-rt (d3d12_draw.inl)
        list_typed[42] = reinterpret_cast<ListTyped>(&list_root_cbv);
        list_typed[50] = reinterpret_cast<ListTyped>(&list_targets);
        list_typed[13] = reinterpret_cast<ListTyped>(&list_draw_indexed);
        if (upstream_compute_root) list_typed[35] = reinterpret_cast<ListTyped>(&list_compute_table_desktop);
        list_typed[36] = reinterpret_cast<ListTyped>(&list_table);
        list_typed[40] = reinterpret_cast<ListTyped>(&list_root_constants);
        list_typed[47] = reinterpret_cast<ListTyped>(&list_index_buffer);
        list_typed[48] = reinterpret_cast<ListTyped>(&list_vertex_buffers);
        install_copy_buffer(list_typed);
        // Upstream ExecuteIndirect (#2232 .. #2262, translated indirect draws with X2W_TRANSLATED_INDIRECT_DRAW=1)
        // replaces the local d3d12_indirect.inl slot 61 only with X2W_UPSTREAM_EXECUTE_INDIRECT=1.
        if (x2w_local::env_on("X2W_UPSTREAM_EXECUTE_INDIRECT"))
            list_typed[x2w_list_slots::execute_indirect_slot] = reinterpret_cast<ListTyped>(&list_execute_indirect_upstream);
        // X2W_D3D12_QUERY_SLOTS=1 (diagnostic): which console methods sit at 28 to 30 is a hypothesis, so the query
        // bindings are opt-in. Unset, the slots stay no-ops.
        if (wchar_t on[4]{}; GetEnvironmentVariableW(L"X2W_D3D12_QUERY_SLOTS", on, 4) && on[0] == L'1') {
            list_typed[28] = reinterpret_cast<ListTyped>(&list_begin_query);
            list_typed[29] = reinterpret_cast<ListTyped>(&list_end_query);
            list_typed[30] = reinterpret_cast<ListTyped>(&list_resolve_query);
        }
    }
    tables[device_kind][16] = reinterpret_cast<void*>(&device_create_root_signature);
    // Translated shader ABI (X2W_GPU_ABI=1): its slot handlers over the ones above (gpu_hooks.inl). Off:
    // nothing changes.
    if (x2w_gpu::gpu_abi_on()) gpu_install();

    // Queue: not the desktop order (slot 18 is GetTimestampFrequency, desktop 16). Slots are
    // dispatched by argument shape until the order is known; see queue_dynamic.
    fill_queue_dynamic(tables[queue_kind], std::make_index_sequence<64>{});
    tables[queue_kind][17] = reinterpret_cast<void*>(&queue_wait);
    tables[queue_kind][18] = reinterpret_cast<void*>(&queue_timestamp_frequency);
    // LOCAL: upstream's slot 31 page batch (#2307) makes much more title memory readable, which multiplies the
    // X2W_MEMORY_SRV_DETILE texture copies past the video memory budget. It is installed only with
    // X2W_UPSTREAM_QUEUE_PAGE_BATCH=1; otherwise slot 31 stays the queue_dynamic no-op (x2w-noop CommandQueue::slot31).
    if (x2w_local::env_on("X2W_UPSTREAM_QUEUE_PAGE_BATCH"))
        tables[queue_kind][31] = reinterpret_cast<void*>(&queue_page_mappings);
    tables[queue_kind][32] = reinterpret_cast<void*>(&queue_present);

    // Fence: GetCompletedValue, SetEventOnCompletion, Signal, GetCreationFlags.
    tables[fence_kind][8] = reinterpret_cast<void*>(&fence_completed_value);
    // Desktop slots 9 (SetEventOnCompletion) and 11 (GetCreationFlags) wait for a confirming call site.
    tables[fence_kind][10] = reinterpret_cast<void*>(&fence_signal);
    tables[fence_kind][12] = reinterpret_cast<void*>(&fence_completed_address);

    tables[allocator_kind][8] = reinterpret_cast<void*>(&allocator_reset);  // Reset

    // Resource: Map, Unmap, GetDesc, GetGPUVirtualAddress, Write/ReadFromSubresource, GetHeapProperties.
    tables[resource_kind][8] = reinterpret_cast<void*>(&resource_map);
    tables[resource_kind][9] = reinterpret_cast<void*>(&resource_unmap);
    tables[resource_kind][10] = reinterpret_cast<void*>(&resource_get_desc);
    tables[resource_kind][11] = reinterpret_cast<void*>(&resource_gpu_address);
    // Desktop Resource 12-14, Heap 8 and DescriptorHeap 8 (GetDesc, struct returns) are
    // not installed until a call site confirms them: the queue's slot 18 showed what a wrong one does.
    (void)&heap_get_desc;
    (void)&descriptor_heap_get_desc;
    tables[descriptor_heap_kind][9] = reinterpret_cast<void*>(&descriptor_heap_cpu_start);
    tables[descriptor_heap_kind][10] = reinterpret_cast<void*>(&descriptor_heap_gpu_start);
    if (device_noop()) fill_all_remaining_noops(std::make_index_sequence<kind_count>{});
    // Say what the tables hold after the fill, so a census run shows which build it ran.
    if (noop_census_enabled())
        log("x2w-noop-census tables: device slot 70 = %s, list slot 12 typed = %s, list slot 44 typed = %s, render mode %d, device noop %d, list noop %d",
            tables[device_kind][70] == reinterpret_cast<void*>(&device_memory_srv) ? "memory SRV handler"
                : tables[device_kind][70] == reinterpret_cast<void*>(&device_noop_slot<70>) ? "NO-OP" : "other",
            list_typed[12] ? "yes" : "NO", list_typed[44] ? "yes" : "NO", render_mode() ? 1 : 0, device_noop() ? 1 : 0, list_noop() ? 1 : 0);
}

SRWLOCK written_lock = SRWLOCK_INIT;
std::unordered_map<u64, TextureWrite>& written_textures() { static std::unordered_map<u64, TextureWrite> m; return m; }

void note_texture_write(void* base, UINT bc7_width, UINT bc7_height)
{
    AcquireSRWLockExclusive(&written_lock);
    auto& written = written_textures()[reinterpret_cast<u64>(base)];
    ++written.generation;
    written.bc7_width = bc7_width;
    written.bc7_height = bc7_height;
    written.desktop_rows = !bc7_width;
    ReleaseSRWLockExclusive(&written_lock);
}
}  // namespace
#include "startup_stubs/d3d12_mem_counters.inl"  // LOCAL memcut: x2w_memory_counters export
#include "startup_stubs/d3d12_memcut_release.inl"  // LOCAL memcut: X2W_MEMCUT_RELEASE
#include "startup_stubs/d3d12_memcut_alias.inl"  // LOCAL memcut: X2W_MEMCUT_ALIAS_SLABS
// LOCAL memcut: upload ring counters for fixtures (d3d12_upload_ring.inl upload_ring_counter).
extern "C" __declspec(dllexport) BOOL x2w_desktop_range_test(UINT64 address, UINT64 bytes, ID3D12Resource** window, UINT64* offset)
{
    if (!window || !offset || !desktop_range(address, bytes, window, offset)) return FALSE;
    (*window)->AddRef();
    return TRUE;
}

extern "C" __declspec(dllexport) UINT64 x2w_gpu_find_address_test(UINT64 address, UINT64 bytes)
{
    GpuFound found{};
    return gpu_find(address, bytes, &found) ? found.resource->GetGPUVirtualAddress() + found.offset : 0;
}
extern "C" __declspec(dllexport) unsigned long long x2w_upload_ring_test(int what) { return upload_ring_counter(what); }
// Translated shader ABI hook counters (x2w_gpu::GpuCounter) for fixtures; -1 for an unknown counter.
extern "C" __declspec(dllexport) long long x2w_gpu_counter(int which) { return gpu_count(which); }

void x2w_startup::attach_device(ID3D12Device* device, void** wrapped)
{
    static const bool ready = (build_tables(), true);
    (void)ready;
    removal_device = device;
    the_device = wrap(device, device_kind);
    *wrapped = the_device;
    prewarm_known_start();  // LOCAL stall: X2W_CX_PREWARM_ALL (d3d12_compute.inl)
}

HRESULT x2w_startup::resolve_placed_build(void* guest_device, UINT64 dest, UINT64 scratch,
    const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO& sizes,
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC& build)
{
    if (build.SourceAccelerationStructureData ||
        (build.Inputs.Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE))
        return E_INVALIDARG;
    constexpr UINT64 alignment = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT;
    const auto translate = [&](UINT64 address, UINT64 bytes, D3D12_RESOURCE_STATES state) -> UINT64 {
        if (address % alignment) return 0;
        const auto found = placed_buffer_range(static_cast<const Wrapper*>(guest_device), address, bytes);
        if (!found.owner || found.reject) return 0;
        const auto* info = resource_info(found.owner);
        if (!info || info->initial_state != state) return 0;
        const auto base = reinterpret_cast<UINT64>(found.owner->own().title_memory);
        const auto gpu = static_cast<ID3D12Resource*>(found.owner->own().real)->GetGPUVirtualAddress();
        const auto offset = address - base;
        if (!gpu || offset > UINT64_MAX - gpu || (gpu + offset) % alignment) return 0;
        return gpu + offset;
    };
    const auto out = translate(dest, sizes.ResultDataMaxSizeInBytes, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
    const auto temp = translate(scratch, sizes.ScratchDataSizeInBytes, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!out || !temp) return E_INVALIDARG;
    build.DestAccelerationStructureData = out;
    build.ScratchAccelerationStructureData = temp;
    return S_OK;
}

extern "C" __declspec(dllexport) BOOL x2w_d3d12_budget_snapshot(x2w_budget::Snapshot* out, SIZE_T bytes)
{
    if (!out || bytes != sizeof(*out) || !x2w_budget::enabled()) return FALSE;
    *out = x2w_budget::snapshot();
    return TRUE;
}

// Called by the XG stand-in after it filled texture memory at a title address.
extern "C" __declspec(dllexport) void x2w_d3d12_texture_written(void* base, unsigned long long bytes)
{
    (void)bytes;
    note_texture_write(base, 0, 0);
}

extern "C" __declspec(dllexport) void x2w_d3d12_bc7_4k_s_written(void* base, UINT width, UINT height)
{
    note_texture_write(base, width, height);
}

bool x2w_d3d12_permissive()
{
    char value[4]{};
    return GetEnvironmentVariableA("X2W_D3D12_PERMISSIVE", value, sizeof(value)) && value[0] == '1';
}

HRESULT x2w_d3d12_permissive_create_device(IUnknown* adapter, const void* parameters, REFIID iid, void** out)
{
    char text[48];
    guid_text(&iid, text, sizeof(text));
    char block[3 * 0x40 + 1];
    hex_text(parameters, 0x40, block, sizeof(block));
    if (!out) return E_POINTER;
    *out = nullptr;
    const bool debug = [] {
        char value[4]{};
        return GetEnvironmentVariableA("X2W_D3D12_DEBUG", value, sizeof(value)) && value[0] == '1';
    }();
    if (debug) {
        ID3D12Debug* layer = nullptr;
        const HRESULT debug_hr = D3D12GetDebugInterface(IID_PPV_ARGS(&layer));
        if (SUCCEEDED(debug_hr)) layer->EnableDebugLayer();
        log("x2w-d3d12 debug layer: 0x%08lx%s", static_cast<unsigned long>(debug_hr),
            SUCCEEDED(debug_hr) ? " enabled" : " (not installed?)");
    }
    // LOCAL stall: X2W_BACKGROUND_PRIORITY=1 (the launcher sets it) lowers this process's GPU scheduling
    // priority class and its CPU priority class to below normal, so the desktop compositor and other
    // programs win while a run saturates the GPU. Both are allowed for an unelevated process lowering its
    // own priority; the results are logged. Unset or 0: today's priorities.
    static const bool background = [] {
        if (!x2w_local::env_on("X2W_BACKGROUND_PRIORITY")) return false;
        // gdi32!D3DKMTSetProcessSchedulingPriorityClass(process, D3DKMT_SCHEDULINGPRIORITYCLASS):
        // 0 idle, 1 below normal, 2 normal (d3dkmthk.h); NTSTATUS result.
        using SetGpuClass = LONG(APIENTRY*)(HANDLE, int);
        const HMODULE gdi = LoadLibraryW(L"gdi32.dll");
        const auto set_gpu = gdi ? reinterpret_cast<SetGpuClass>(GetProcAddress(gdi, "D3DKMTSetProcessSchedulingPriorityClass")) : nullptr;
        const LONG gpu = set_gpu ? set_gpu(GetCurrentProcess(), 1) : -1;
        const BOOL cpu = SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
        const DWORD cpu_error = cpu ? 0 : GetLastError();
        log("x2w-d3d12 background priority: GPU scheduling class below normal -> 0x%08lx%s; CPU priority class below normal -> %s "
            "(error %lu, class now 0x%lx)", static_cast<unsigned long>(gpu), set_gpu ? "" : " (entry point not found)",
            cpu ? "ok" : "refused", cpu_error, GetPriorityClass(GetCurrentProcess()));
        return true;
    }();
    (void)background;
    if (x2w_local::env_on("X2W_D3D12_DRED")) {
        // LOCAL: device-removed extended data (breadcrumbs and page fault), reported by check_device.
        ID3D12DeviceRemovedExtendedDataSettings* dred = nullptr;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dred)))) {
            dred->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            dred->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            dred->Release();
            log("x2w-d3d12 DRED enabled");
        }
    }
    ID3D12Device* device = nullptr;
    const HRESULT hr = kind_for(iid) == device_kind || iid == __uuidof(IUnknown)
        ? D3D12CreateDevice(adapter ? reinterpret_cast<IUnknown*>(unwrap(reinterpret_cast<u64>(adapter))) : nullptr,
                            D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device))
        : E_NOINTERFACE;
    log("x2w-d3d12 D3D12XboxCreateDevice adapter=%p params=[%s] riid=%s -> 0x%08lx build=%s", static_cast<void*>(adapter),
        block, text, static_cast<unsigned long>(hr), x2w_build_rev);
    if (FAILED(hr)) return hr;
    if (debug) device->QueryInterface(IID_PPV_ARGS(&info_queue));
    x2w_startup::attach_device(device, out);
    return S_OK;
}
