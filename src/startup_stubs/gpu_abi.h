// Shared switches, ABI constants and types for running translated DXIL. These types depend only on Windows, D3D12
// and standard headers. The Wrapper-dependent hooks live in gpu_hooks.h inside the device shim's anonymous
// namespace.
//
// Shader keys are the first eight BLAKE3 bytes of the exact title bytecode, formatted as 16 lowercase hex digits.
// GpuStage describes the requested pipeline stage and GpuHwStage describes the translated hardware stage.
// The reader converts the serialized root codes (0..5 or 6+x) to GpuRootType and a separate code for Other.
#pragma once

#include <windows.h>
#include <d3d12.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace x2w_depth_twins { struct Twins; }
namespace x2w_gpu {

// ---------------------------------------------------------------------------------------------------------------
// Switches. X2W_GPU_ABI=1 turns the whole path on (read once). Off (unset, the default): every call site runs
// the old X2W path exactly; that is the rollback. On: the translated shader path only, no mixing with gfx2hlsl
// pipelines. The sub-switches are on by default under ABI=1 and turned off with "=0" (for bisecting); off when ABI
// is off.
// ---------------------------------------------------------------------------------------------------------------
inline bool gpu_env_is(const char* name, char first) noexcept
{
    const DWORD saved = GetLastError();
    char value[4]{};
    const bool is = GetEnvironmentVariableA(name, value, sizeof(value)) && value[0] == first;
    SetLastError(saved);
    return is;
}
inline bool gpu_abi_on() noexcept
{
    static const bool on = gpu_env_is("X2W_GPU_ABI", '1');
    return on;
}
// X2W_GPU_INDIRECT=0: ExecuteIndirect calls are dropped and counted (WP4).
inline bool gpu_indirect_on() noexcept
{
    static const bool on = gpu_abi_on() && !gpu_env_is("X2W_GPU_INDIRECT", '0');
    return on;
}
// X2W_GPU_MESH=0: mesh pipelines are refused (WP2) and mesh dispatches dropped (WP4).
inline bool gpu_mesh_on() noexcept
{
    static const bool on = gpu_abi_on() && !gpu_env_is("X2W_GPU_MESH", '0');
    return on;
}
// X2W_GPU_TWINS=0: no raw twins in the mirror heap (WP5). Must agree with the export manifest's raw_views
// (GpuManifest::raw_views == 0 for an export made with raw_views=0); the reader asserts it.
inline bool gpu_twins_on() noexcept
{
    static const bool on = gpu_abi_on() && !gpu_env_is("X2W_GPU_TWINS", '0');
    return on;
}
// X2W_GPU_KEYLOG=<path> (G0): every bytecode at device slots 10/11/62/86 is logged once per (key, stage) as
// "<key16> <stage> slot<N> <bytes>" (gpu_hooks.inl). Unset: no key file (counters still run).

// ABI constants embedded in the translated DXIL.
namespace abi {
// Binding unit: the console's. A table's user-data dword is (handle - heap start) in these bytes.
constexpr UINT view_increment = 32;     // CBV_SRV_UAV, device slot 15 under ABI
constexpr UINT sampler_increment = 16;  // SAMPLER, device slot 15 under ABI
// RTV and DSV retain the desktop increment for handle mapping.

// Register spaces of the translated shaders.
constexpr UINT user_data_space = 0;  // b0: the per-stage 32-bit root constants (user data)
constexpr UINT sampler_space = 0;    // s<n>: immediate samplers as static samplers
constexpr UINT root_cbv_space = 1;   // b<param> / t<param> / u<param>: root CBV/SRV/UAV, param = title root index

// Reserved shader runtime slots in root_cbv_space.
constexpr std::uint8_t slot_strides = 255;  // root SRV: SLOT_STRIDES buffer (dword i = stride of view slot i)
constexpr std::uint8_t slot_memory = 254;   // root SRV: title memory directory (BVH only; such shaders are dropped)
constexpr std::uint8_t slot_faults = 253;   // root UAV: memory faults (BVH only; dropped)
constexpr std::uint32_t slot_strides_count = 1u << 21;  // dwords of the SLOT_STRIDES buffer
constexpr std::uint32_t view_stride_bits = (1u << 30) - 1;
constexpr std::uint32_t view_integer = 1u << 31;  // typed buffer view of a UINT/SINT format
constexpr std::uint32_t view_signed = 1u << 30;   // with view_integer: SINT
constexpr std::uint32_t memory_pages = 1u << 21;
constexpr std::uint32_t memory_read_only = 1u << 31;
constexpr std::uint32_t memory_faults = 4096;

// RootType::Other codes of user-data dwords the host computes (GpuUserData::other).
constexpr std::uint32_t user_memory_base = 0x100;
constexpr std::uint32_t user_view_address = 0x101;  // +1: high bits (0x102)
constexpr std::uint32_t user_view_bias = 0x103;
constexpr std::uint32_t user_root_address = 0x104;  // +1: high bits (0x105)
constexpr std::uint32_t user_dispatch_threads_x = 0x106;

// Mirror view heap layout (WP5), with twins: [0, count) title slots, [count, null) dynamic, null slot, twins.
constexpr std::uint32_t raw_views = 1u << 21;           // twin of slot i at raw_views + i
constexpr std::uint32_t view_heap_limit = (1u << 21) - 1;  // the null slot; view indices are clamped to it
constexpr std::uint32_t heap_views_with_twins = 1u << 22;
constexpr std::uint32_t dynamic_min = 1u << 18;  // dynamic_slots() = clamp(null - count, dynamic_min, dynamic_max)
constexpr std::uint32_t dynamic_max = 1u << 20;

// Console pipeline-stream subobject numbers that differ from the PC's.
constexpr std::uint32_t subobject_amplification = 23;
constexpr std::uint32_t subobject_mesh = 24;

// Console command-signature argument type run as a plain dispatch.
constexpr std::uint32_t indirect_console_dispatch = 0x69;
}  // namespace abi

// Stage of the supplied bytecode. Device slot 10 orders vertex, pixel, domain, hull and geometry in the head. Slots
// 11 and 62 take compute. Slot 86 takes stream subobjects 1–6, 23 and 24.
enum class GpuStage : std::uint8_t { vertex, pixel, domain, hull, geometry, compute, amplification, mesh, count };
inline const char* gpu_stage_name(GpuStage s) noexcept
{
    static const char* const names[] = {"vertex", "pixel", "domain", "hull", "geometry", "compute", "amplification", "mesh"};
    return s < GpuStage::count ? names[static_cast<int>(s)] : "unknown";
}
// Console hardware stage of the translated program, in cache order.
enum class GpuHwStage : std::uint8_t { ls, hs, es, gs, vs, ps, ms, cs };

// Exported shader and pipeline metadata. Spans point into the process-lifetime database mapping. Counts are in
// elements.
template <typename T>
struct GpuSpan {
    const T* data = nullptr;
    std::uint32_t count = 0;
    const T* begin() const noexcept { return data; }
    const T* end() const noexcept { return data + count; }
    const T& operator[](std::uint32_t i) const noexcept { return data[i]; }
    bool empty() const noexcept { return count == 0; }
};

// Decoded title root parameter kind.
enum class GpuRootType : std::uint8_t {
    constants,          // a root-constant dword
    descriptor,         // a 64-bit buffer address (root CBV/SRV/UAV)
    table,              // the heap address of a descriptor table range: (handle - heap start) in console bytes
    immediate_sampler,  // a sampler the root signature fixes
    resource_heap,      // the CBV/SRV/UAV heap's address (SM 6.6): 0 on the host
    sampler_heap,       // the sampler heap's address: 0 on the host
    other,              // `other` holds the code: abi::user_* (host-computed dwords), or an unknown value
};

// User-data dword `index` holds dword `dword` of title root parameter `slot`.
struct GpuUserData {
    std::uint32_t index;
    std::uint32_t slot;  // title root parameter index
    GpuRootType root;
    std::uint32_t other;  // RootType::Other code when root == other, else 0
    std::uint32_t dword;  // dword within the parameter (for user_dispatch_threads_x: the group width)
};

// The four console sampler words of an immediate sampler s<n> (HostShader::samplers).
struct GpuSamplerWords {
    std::uint32_t words[4];
};

// Decoded immediate sampler in D3D12 enum values. The root builder applies filter_for_host because anisotropic
// point-mip needs D3D12_OPTIONS19.
struct GpuStaticSampler {
    std::uint32_t filter;
    std::uint32_t address[3];  // U, V, W
    float mip_lod_bias;
    std::uint32_t max_anisotropy;
    std::uint32_t comparison;
    std::uint32_t border;
    float min_lod;
    float max_lod;
};

// A heap descriptor whose byte address is evaluated at bind time from a table base and a root constant.
struct GpuRootIndexedView {
    std::uint32_t table_slot;     // u8
    std::uint32_t constant_slot;  // u8
    std::uint32_t constant_dword;
    std::uint32_t byte_offset;
    std::uint32_t shift;  // u8
};

// Srv=0, Sampler=1, Cbv=2, Uav=3, Other(n) (kind == other, `other_kind` = n).
enum class GpuResourceType : std::uint8_t { srv, sampler, cbv, uav, other };

// A descriptor range the console program reads (Translated::used). All eight fields.
struct GpuUsedResource {
    GpuResourceType kind;
    std::uint8_t other_kind;  // ResourceType::Other(n) when kind == other
    GpuRootType root;
    std::uint32_t root_other;  // RootType::Other code when root == other
    std::uint32_t table_offset_dwords;
    std::uint32_t lower_bound;  // shader register of the first descriptor read
    std::uint32_t stride;
    std::uint8_t dwords;     // descriptor size, 4 or 8
    std::uint8_t user_data;  // user SGPR
    std::uint8_t slot;       // title root parameter
};

struct GpuShader {
    char key[17];                   // C1 key, NUL-terminated
    GpuStage stage;             // the stage the export recorded it under
    GpuHwStage hw_stage;        // HostShader::stage
    const void* dxil = nullptr;     // the DXIL container (signed/hashed as D3D12 expects)
    std::size_t dxil_bytes = 0;
    std::uint16_t material_fields = 0;
    GpuSpan<GpuUserData> user_data;
    GpuSpan<GpuSamplerWords> samplers;          // s<n> at position n
    GpuSpan<GpuStaticSampler> static_samplers;  // the same, decoded; s<n> at position n
    GpuSpan<std::uint8_t> root_cbvs;  // title root parameter indices: b<i> space1
    GpuSpan<std::uint8_t> root_srvs;  // t<i> space1 (may contain abi::slot_strides / slot_memory)
    GpuSpan<std::uint8_t> root_uavs;  // u<i> space1 (may contain abi::slot_faults)
    GpuSpan<GpuRootIndexedView> root_indexed_views;
    // Translated (not in the cache record; the export must add them, C2).
    bool has_threads = false;  // Translated::threads is Some
    std::uint32_t threads[3]{};  // compute threads per group
    GpuSpan<GpuUsedResource> used;
    bool stores = true;             // the code can store to buffer memory (true when unknown)
    bool stores_by_address = true;  // the code can store through raw addresses (true when unknown)
    bool builder = false;

    // HostShader::user_data_dwords: round4(max index + 1), 0 when it reads none. Root constants must cover it.
    std::uint32_t user_data_dwords() const noexcept
    {
        std::uint32_t n = 0;
        for (const auto& u : user_data)
            if (u.index + 1 > n) n = u.index + 1;
        return (n + 3) & ~3u;
    }
};

// Contract C3: the export manifest header.
struct GpuManifest {
    std::uint32_t view_heap_limit;  // expected abi::view_heap_limit
    std::uint32_t raw_views;        // expected abi::raw_views (0: exported without twins)
    bool host_rdna;
    std::uint32_t max_wave;
    std::uint64_t opt_bits;  // X2W_SHADERCFG_SHADER_OPT
    bool group_sync;         // X2W_SHADERCFG_GROUP_SYNC
    bool helper_own;         // X2W_SHADERCFG_HELPER_OWN
    std::uint64_t bvh_bits;  // BVH switches
    char commit[48];         // translator revision recorded at export
};

// Pipeline creation records title bytecode and state. The host pipeline is built on first use. A draw receives null
// while its pipeline is pending or after refusal.
//
// The creation hook computes the key before the title bytecode goes out of scope.
struct GpuStageCode {
    GpuStage stage;
    const void* bytes;  // title memory; valid during gpu_create_pipeline only
    std::size_t size;
    char key[17];
};

// What the creation slot passed (WP0 fills it, WP2 records it).
struct GpuCreateDesc {
    int slot;                // device slot: 10, 11, 62 or 86
    const void* desc;        // the title's description: GraphicsHead (10), ComputeHead (11/62), stream bytes (86)
    std::size_t desc_bytes;  // slot 86: the stream's SizeInBytes; else the head's size
    const void* title_root;  // the root signature wrapper the head names (10/11/62), or the stream's (86), or null
    std::uint32_t stage_count;
    GpuStageCode stages[8];
    bool unreadable;  // a stage's bytes, or the stream, could not be read (the pipeline will be dropped)
};

// The host root parameter a title root view is bound at.
struct GpuRootSlot {
    std::uint8_t title;  // title root parameter index
    std::uint8_t host;   // host root parameter index (a one-view table by default, or a root descriptor)
    bool table;          // true: a one-view descriptor table (bounded_root_srvs); false: a root descriptor
};

struct GpuPipelineStage {
    const GpuShader* shader = nullptr;
    GpuStage stage{};
    std::uint32_t user_data_param = UINT32_MAX;  // host root parameter of the b0 constants, UINT32_MAX: none
    std::uint32_t user_data_dwords = 0;
    std::vector<GpuRootSlot> root_cbvs, root_srvs, root_uavs;
};

struct GpuInputElement {
    std::uint32_t slot;
    std::uint32_t offset;
    DXGI_FORMAT format;
    bool per_instance;
    std::uint32_t step_rate;
};

// A made pipeline, owned by WP2, read by WP3/WP4. The title wrapper's own().real stays null: X2W's old
// list_pipeline must never bind it.
struct GpuPipeline {
    bool compute = false;
    bool mesh = false;
    bool rects = false;  // Drawn with the rectangle geometry shader
    ID3D12PipelineState* pso = nullptr;
    UINT color_mask = D3D12_COLOR_WRITE_ENABLE_ALL;
    bool depth_test = false, depth_write = false;
    ID3D12RootSignature* root = nullptr;
    std::uint32_t stage_count = 0;
    GpuPipelineStage stages[4];  // compute: CS; mesh: (AS) MS PS; graphics: VS PS
    std::uint32_t vertex_slots = 0;  // mask of vertex buffer slots the input layout reads
    std::vector<GpuInputElement> inputs;
    D3D12_PRIMITIVE_TOPOLOGY_TYPE topology_type = D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED;
    std::uint32_t title_topology = 0;  // the stream's topology as the title gave it (console values)
    std::uint32_t strides_param = UINT32_MAX;  // host param of SLOT_STRIDES, UINT32_MAX: none
    // DispatchX (WP4): where USER_DISPATCH_THREADS_X lives, UINT32_MAX: none.
    std::uint32_t threads_x_param = UINT32_MAX;
    std::uint32_t threads_x_dword = 0;
    void* impl = nullptr;  // WP2's private state
};

// The targets bound at the draw, for pipelines whose stream formats differ from the bound views (WP2 makes a variant).
struct GpuTargets {
    std::uint32_t count = 0;
    DXGI_FORMAT rtv[8]{};
    DXGI_FORMAT dsv = DXGI_FORMAT_UNKNOWN;
    std::uint32_t samples = 1;
    bool read_depth_plane = false, read_stencil_plane = false;
};

// ---------------------------------------------------------------------------------------------------------------
// Root arguments (WP3). One per title root parameter, graphics and compute kept apart, any number of parameters and
// dwords. WP3's per-list record (GpuListState, gpu_roots.h) holds vectors of these.
// ---------------------------------------------------------------------------------------------------------------
enum class GpuRootKind : std::uint8_t { none, table, constants, cbv, srv, uav };
struct GpuRoot {
    GpuRootKind kind = GpuRootKind::none;
    std::uint64_t handle = 0;   // table: the title GPU handle
    std::uint64_t address = 0;  // cbv/srv/uav: the title GPU address
    std::vector<std::uint32_t> dwords;  // constants
};
struct GpuListState;  // WP3, gpu_roots.h

// ---------------------------------------------------------------------------------------------------------------
// Mirror heaps and views (WP5). Opaque heap; the functions are in gpu_hooks.h.
// ---------------------------------------------------------------------------------------------------------------
struct GpuHostHeap;  // WP5

// One title shader-visible heap as WP3 needs it.
struct GpuHeapInfo {
    GpuHostHeap* heap = nullptr;
    const void* title_heap = nullptr;  // the title's descriptor heap wrapper
    std::uint64_t title_gpu = 0;  // the GPU start the title was given (handles are title_gpu + index * increment)
    std::uint64_t title_cpu = 0;  // the title's descriptor memory
    std::uint32_t count = 0;      // title slots
    bool sampler = false;
    ID3D12DescriptorHeap* host = nullptr;  // for SetDescriptorHeaps
};

// View decoded from the title slot's token or descriptor words.
enum class GpuViewKind : std::uint8_t { none, srv, uav, cbv, sampler };
struct GpuViewInfo {
    GpuViewKind kind = GpuViewKind::none;
    bool texture = false;
    bool buffer = false;
    UINT first_mip{}, first_slice{}, plane{};
    DXGI_FORMAT view_format = DXGI_FORMAT_UNKNOWN;
    const void* title_resource = nullptr;  // the resource wrapper, or null (placed/address views)
    ID3D12Resource* desktop = nullptr;     // the desktop resource the view names, or null
    std::shared_ptr<x2w_depth_twins::Twins> depth_twins;
    std::uint64_t title_address = 0;       // buffer: the title GPU address of the first byte
    std::uint64_t bytes = 0;               // buffer: the view's size
    std::uint32_t stride = 0;              // buffer: bytes per element (the SLOT_STRIDES value)
};

// ---------------------------------------------------------------------------------------------------------------
// Title memory (WP3, gpu_find.h): what is at a title GPU address, bounded (never the bare window end).
// ---------------------------------------------------------------------------------------------------------------
struct GpuFound {
    ID3D12Resource* resource = nullptr;  // the desktop window or buffer
    std::uint64_t offset = 0;            // of `address` in it
    std::uint64_t bytes_left = 0;        // from `address` to the end of the title buffer or the contiguous run
};

// ---------------------------------------------------------------------------------------------------------------
// Counters kept by the WP0 hooks (not by the stubs, so they stay when the stubs are filled). Read by tests through
// the dll export x2w_gpu_counter(int) (gpu_hooks.inl) and logged as "x2w-gpu ..." lines.
// ---------------------------------------------------------------------------------------------------------------
enum GpuCounter : int {
    vc_pipelines,       // creation calls at 10/11/62/86 under ABI
    vc_pipelines_kept,  // gpu_create_pipeline returned true
    vc_keys,            // distinct (key, stage) pairs logged
    vc_refused_57,
    vc_refused_67,
    vc_heaps,           // descriptor heaps passed to gpu_heap_created
    vc_views,           // view slots passed to gpu_heap_view
    vc_copies,          // descriptor copies passed to gpu_heap_copy*
    vc_setters,         // list setter calls passed on
    vc_draws, vc_draws_dropped,
    vc_dispatches, vc_dispatches_dropped,
    vc_meshes, vc_meshes_dropped,
    vc_indirects, vc_indirects_dropped,
    vc_mesh_x_dropped,  // DispatchMeshX (dropped)
    vc_resets, vc_closes, vc_submits,
    vc_count
};

}  // namespace x2w_gpu
