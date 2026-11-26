#pragma once

// A DXIL vertex, pixel, compute or mesh shader under construction: the module, the dx.op declarations it
// calls, its signatures and bound resources, and the metadata and container that wrap it up (SM 6.6,
// validator 1.6).

#include <array>
#include <atomic>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "x2w_shader/emit/bitcode.h"
#include "x2w_shader/emit/container.h"

namespace x2w::vs::shader {

using bitcode::Attr;
using bitcode::Md;
using bitcode::Module;
using bitcode::Ty;
using bitcode::V;
using container::SigElement;

// Where the shader reads its user data: b0 in this space.
inline constexpr std::uint32_t USER_DATA_SPACE = 0;
// Where immediate samplers are bound: s<n> in this space.
inline constexpr std::uint32_t SAMPLER_SPACE = 0;
// Where root CBVs and SRVs are bound: b<root parameter index> and t<root parameter index> in
// this space.
inline constexpr std::uint32_t ROOT_CBV_SPACE = 1;

// The root SRV t<SLOT_STRIDES> in ROOT_CBV_SPACE of a shader that reads buffers through
// descriptors it picks from the view heap at an index it computes (SM 6.6 dynamic resources): a
// raw buffer of SLOT_STRIDES_COUNT dwords the host keeps, dword i the bytes per element of the
// buffer view at heap slot i (0: none or not a buffer). A shader that has this root SRV also
// tells the host that the buffer views of the whole heap must be ready when it runs.
inline constexpr std::uint8_t SLOT_STRIDES = 255;
// The bits of a SLOT_STRIDES dword that hold the stride.
inline constexpr std::uint32_t VIEW_STRIDE_BITS = (1u << 30) - 1;
// In a SLOT_STRIDES dword: a typed buffer view of an integer format (UINT or SINT), whose 16-bit
// typed accesses (_d16) pass bits.
inline constexpr std::uint32_t VIEW_INTEGER = 1u << 31;
// With VIEW_INTEGER: a signed (SINT) format.
inline constexpr std::uint32_t VIEW_SIGNED = 1u << 30;
// Dwords of the SLOT_STRIDES buffer: heap indices are clamped to it.
inline constexpr std::uint32_t SLOT_STRIDES_COUNT = 1u << 21;

// Title memory by address (resource MEMORY): The root SRV t<SLOT_MEMORY> in ROOT_CBV_SPACE is a directory
// of MEMORY_PAGES entries of two dwords, one per 64 KiB of title address space (host copy number,
// MEMORY_READ_ONLY; byte offset in that copy); copy n is the raw view at view-heap slot base + n (base: the
// user-data dword USER_MEMORY_BASE); the root UAV u<SLOT_FAULTS> takes the page numbers of accesses that
// found no copy (MEMORY_FAULTS dwords, dword 0 a sink).
inline constexpr std::uint8_t SLOT_MEMORY = 254;
inline constexpr std::uint8_t SLOT_FAULTS = 253;
// Entries of the SLOT_MEMORY directory: title addresses below 128 GiB.
inline constexpr std::uint32_t MEMORY_PAGES = 1u << 21;
// In a directory entry's first dword: shaders do not store there by address.
inline constexpr std::uint32_t MEMORY_READ_ONLY = 1u << 31;
// Dwords of the SLOT_FAULTS buffer.
inline constexpr std::uint32_t MEMORY_FAULTS = 4096;
// RootType::Other code of a user-data dword the host computes: the view heap slot of the memory
// copies' views (SLOT_MEMORY).
inline constexpr std::uint32_t USER_MEMORY_BASE = 0x100;
// RootType::Other code: the low 32 bits of the title address of the buffer view at dword `dword`
// of the descriptor table of root parameter `slot`; USER_VIEW_ADDRESS + 1: its bits from 32 up.
inline constexpr std::uint32_t USER_VIEW_ADDRESS = 0x101;
// RootType::Other code: the bytes (0 to 15) by which the raw view the host binds for the root
// view of root parameter `slot` begins before the title's address. Asked for every root UAV.
inline constexpr std::uint32_t USER_VIEW_BIAS = 0x103;
// RootType::Other code: the low 32 bits of the title address in the root view of root parameter
// `slot`; USER_ROOT_ADDRESS + 1: its bits from 32 up.
inline constexpr std::uint32_t USER_ROOT_ADDRESS = 0x104;
// RootType::Other code of a user-data dword the host sets at a dispatch counted in threads (the
// console's DispatchX): how many of the threads numbered along X run (0: all of every group).
inline constexpr std::uint32_t USER_DISPATCH_THREADS_X = 0x106;

// ShaderFlags bits (metadata) and their SFI0 feature bits.
namespace flags {
inline constexpr std::uint64_t RAW_BUFFERS = 1ull << 4;
inline constexpr std::uint64_t DOUBLES = 1ull << 2;
inline constexpr std::uint64_t STENCIL_REF = 1ull << 11;
inline constexpr std::uint64_t TYPED_UAV_LOADS = 1ull << 13;
// Unordered views in a vertex, geometry or mesh program.
inline constexpr std::uint64_t UAVS_EVERY_STAGE = 1ull << 16;
inline constexpr std::uint64_t ROVS = 1ull << 18;
inline constexpr std::uint64_t WAVE_OPS = 1ull << 19;
inline constexpr std::uint64_t BARYCENTRICS = 1ull << 22;
inline constexpr std::uint64_t INT64 = 1ull << 20;
inline constexpr std::uint64_t ATOMIC_TYPED64 = 1ull << 27;
inline constexpr std::uint64_t ATOMIC_HEAP64 = 1ull << 32;
inline constexpr std::uint64_t RESOURCE_HEAP = 1ull << 30;
inline constexpr std::uint64_t SAMPLER_HEAP = 1ull << 31;
}  // namespace flags

struct Types {
    Ty void_;
    Ty i1;
    Ty i32;
    Ty f32;
    Ty handle;
    Ty props;
    Ty bind;
    Ty cbuf_ret;
    Ty res_ret;
};

// What a mesh shader declares beyond its thread-group size.
struct Mesh {
    std::uint32_t max_vertices = 0;
    std::uint32_t max_primitives = 0;
    // 1 lines, 2 triangles.
    std::uint8_t topology = 0;
};

// What a geometry shader declares: triangles in, triangle strips out on stream 0, one instance.
struct Geometry {
    std::uint32_t max_vertices = 0;
};

// The null slot every host view heap keeps at its end (0: no limit). Shaders translated while it
// is set read that slot for any view index at or past it.
void set_view_heap_limit(std::uint32_t null_slot);
std::uint32_t view_heap_limit();

// Where the host's raw twins of its buffer views start in its view heaps (0: there are none). A power of
// two above every slot index. See RAW_VIEWS.
void set_raw_views(std::uint32_t first);
std::uint32_t raw_views();
// Override raw_views() on the current test thread.
void set_raw_views_here(std::optional<std::uint32_t> first);

struct Shader {
    Module m;
    Types t;
    bool pixel = false;
    std::vector<SigElement> inputs;
    std::vector<SigElement> outputs;
    // A mesh shader's per-primitive outputs.
    std::vector<SigElement> prims;
    // A mesh shader (threads is its thread-group size).
    std::optional<Mesh> mesh;
    // A geometry shader (the rectangle stage, rect.cpp); pixel is false.
    std::optional<Geometry> geometry;
    // Dwords of user data the shader reads (0: no user-data buffer).
    std::uint32_t user_data = 0;
    // Immediate samplers, bound as s0.. in SAMPLER_SPACE.
    std::uint32_t samplers = 0;
    // Bit n: immediate sampler n is a comparison sampler.
    std::uint64_t cmp_samplers = 0;
    // Root CBVs read, by root parameter index, with their handles.
    std::vector<std::pair<std::uint8_t, V>> root_cbvs;
    // Root SRVs (raw buffers) read, likewise.
    std::vector<std::pair<std::uint8_t, V>> root_srvs;
    // Root UAVs (raw buffers) read or written, likewise.
    std::vector<std::pair<std::uint8_t, V>> root_uavs;
    std::uint64_t flags = 0;
    // Heap handles made with a non-uniform index.
    std::uint32_t nonuniform_handles = 0;
    // A compute shader with this thread-group size.
    std::optional<std::array<std::uint32_t, 3>> threads;
    // The wave size a compute shader requires (WaveSize).
    std::optional<std::uint32_t> wave_size;
    // Thread-group shared memory, in dwords.
    std::uint32_t lds_dwords = 0;
    std::size_t entry = 0;

    explicit Shader(bool pixel);

    // The thread-group shared memory: [lds_dwords x i32] in address space 3.
    V lds();
    // Calls dx.op.<name> (class and overload) with opcode and args.
    V op(std::size_t block, std::string_view name, Attr attr, Ty ret, std::uint32_t opcode,
         std::span<const V> args);
    V op(std::size_t block, std::string_view name, Attr attr, Ty ret, std::uint32_t opcode,
         std::initializer_list<V> args) {
        return op(block, name, attr, ret, opcode, std::span<const V>(args.begin(), args.size()));
    }
    // The handle of the user-data constant buffer, made in the entry block.
    V user_data_handle();
    // The handle of the root CBV of root parameter slot, made in the entry block.
    V root_cbv_handle(std::uint8_t slot);
    // The handle of the root SRV (a raw buffer) of root parameter slot.
    V root_srv_handle(std::uint8_t slot);
    // The handle of the root UAV (a raw buffer) of root parameter slot: u<slot> in ROOT_CBV_SPACE.
    V root_uav_handle(std::uint8_t slot);
    // The handle of immediate sampler n (a comparison sampler when its bit in cmp_samplers is set).
    V sampler_handle(std::size_t block, std::uint32_t n);
    V annotate(std::size_t block, V h, std::uint32_t props0, std::uint32_t props1);
    // A descriptor heap handle, annotated with props; nonuniform when the index may differ
    // between lanes (NonUniformResourceIndex).
    V heap_handle(std::size_t block, V index, bool sampler, bool nonuniform, std::pair<std::uint32_t, std::uint32_t> props);
    // heap_handle; bounded: the index is inside the host's heap already (raw_views).
    V heap_handle_at(std::size_t block, V index, bool sampler, bool nonuniform,
                     std::pair<std::uint32_t, std::uint32_t> props, bool bounded);
    // Finish the shader container. Call once.
    std::vector<std::uint8_t> finish();

private:
    std::optional<V> lds_;
    V main_;
    void fix_user_data_size(std::uint32_t size);
    std::optional<Md> signature_md(std::vector<SigElement> elements);
};

}  // namespace x2w::vs::shader
