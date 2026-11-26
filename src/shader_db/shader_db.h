// Precompiled shader database. Each record is keyed by the stage and the first eight bytes of BLAKE3 over the exact
// D3D12_SHADER_BYTECODE passed at pipeline creation, read as a big-endian u64. The FNV hash links the record to
// request files.
//
// The little-endian file contains a 64-byte VxHeader, VxIndex entries sorted by key, VxRecord entries, element
// arrays, DXIL containers and manifest text. Offsets are relative to the start of the file. Each record includes
// the translated shader and its thread dimensions, resource use and store flags. The reader rejects incompatible
// manifest settings, supports concurrent lookups after opening, and logs each missing key once.
#pragma once

#include <cstddef>
#include <cstdint>

namespace shader_db {

// Hardware stage order in the serialized shader cache.
enum class Stage : uint32_t { ls = 0, hs = 1, es = 2, gs = 3, vs = 4, ps = 5, ms = 6, cs = 7, any = 0xffffffffu };

// Serialized root kinds: 0 Constants, 1 Descriptor, 2 Table, 3 ImmediateSampler, 4 ResourceHeap, 5 SamplerHeap, 6 +
// x Other(x).
enum RootKind : uint32_t { root_constants = 0, root_descriptor = 1, root_table = 2, root_immediate_sampler = 3,
                           root_resource_heap = 4, root_sampler_heap = 5, root_other_base = 6 };
// Resource kinds of `used`: 0 Srv, 1 Sampler, 2 Cbv, 3 Uav, 4 + x Other(x).
enum ResourceKind : uint32_t { resource_srv = 0, resource_sampler = 1, resource_cbv = 2, resource_uav = 3, resource_other_base = 4 };

#pragma pack(push, 1)
struct VxHeader {
    char magic[8];  // "VXDB0001"
    uint32_t version, record_count;
    uint64_t index_offset, manifest_offset;
    uint32_t manifest_bytes, reserved;
    uint64_t file_bytes;
    uint8_t pad[16];
};
struct VxIndex { uint64_t key, record_offset; };
struct VxUserData { uint32_t index, slot, root, dword; };  // dword `index` of b0 holds root parameter `slot`'s dword
struct VxStaticSampler {
    uint32_t filter, address[3];
    float mip_lod_bias;
    uint32_t max_anisotropy, comparison, border;
    float min_lod, max_lod;
};
struct VxRootIndexedView { uint32_t table_slot, constant_slot, constant_dword, byte_offset, shift; };
struct VxUsed { uint32_t kind, root, table_offset_dwords, lower_bound, stride, dwords, user_data, slot; };
enum : uint32_t { flag_stores = 1, flag_stores_by_address = 2, flag_builder = 4, flag_threads = 8 };
struct VxRecord {
    uint64_t key, fnv64;
    uint32_t stage, container_bytes;
    uint64_t dxil_offset;
    uint32_t dxil_bytes, user_data_dwords;
    uint32_t user_data_count, sampler_count, static_sampler_count, root_cbv_count;
    uint32_t root_srv_count, root_uav_count, root_indexed_view_count, used_count;
    uint64_t user_data_offset, samplers_offset, static_samplers_offset, root_cbvs_offset;
    uint64_t root_srvs_offset, root_uavs_offset, root_indexed_views_offset, used_offset;
    uint32_t threads[3], flags, wave_size, vgprs;
};
#pragma pack(pop)
static_assert(sizeof(VxHeader) == 64, "header layout");
static_assert(sizeof(VxRecord) == 160, "record layout");
static_assert(sizeof(VxStaticSampler) == 40, "static sampler layout");

// A record with its arrays resolved (pointers into the mapped file; valid until gpu_db_close).
struct GpuShader {
    uint64_t key, fnv64;
    Stage stage;
    uint32_t container_bytes;
    const void* dxil;
    std::size_t dxil_bytes;
    uint32_t user_data_dwords;  // root constants of the stage must cover this many dwords (b0, space 0)
    const VxUserData* user_data; uint32_t user_data_count;
    const uint32_t (*samplers)[4]; uint32_t sampler_count;  // immediate samplers s<n>, console words
    const VxStaticSampler* static_samplers; uint32_t static_sampler_count;  // the same decoded, s<n> space 0
    const uint8_t* root_cbvs; uint32_t root_cbv_count;  // root parameter indices, b<i> in ROOT_CBV_SPACE
    const uint8_t* root_srvs; uint32_t root_srv_count;  // t<i> in ROOT_CBV_SPACE (raw buffers)
    const uint8_t* root_uavs; uint32_t root_uav_count;  // u<i> in ROOT_CBV_SPACE (raw buffers)
    const VxRootIndexedView* root_indexed_views; uint32_t root_indexed_view_count;
    const VxUsed* used; uint32_t used_count;
    bool has_threads; uint32_t threads[3];
    bool stores, stores_by_address, builder;
    uint32_t wave_size, vgprs;  // of the console program (derived)
};

// The settings the export was made for (manifest.txt), as numbers; 0 when absent.
struct Manifest {
    uint32_t max_wave, view_heap_limit, raw_views, host_rdna, loop_limit, group_sync, shader_opt;
    // The X2W_SHADERCFG_* switches in the translation's key, as the export saw them (empty: unset).
    char helper_own[16], bvh_fences[16], bvh_bounds[16];
    char build[65], commit[64], producer[32];
};

using MissLog = void (*)(const char* line);

// Maps `path`. Refuses (returns false, logs why) a file that is not a database, or whose manifest's view_heap_limit
// or raw_views differ from the expected values (pass 0 to skip a check). Idempotent for one path.
bool gpu_db_open(const wchar_t* path, uint32_t expected_view_heap_limit, uint32_t expected_raw_views, MissLog log);
void gpu_db_close();
const Manifest* gpu_db_manifest();
std::size_t gpu_db_count();

// C1 key of the bytes (BLAKE3, first 8 bytes big-endian).
uint64_t gpu_key(const void* bytes, std::size_t n);
// C5: the record for these bytes, or null (C6: the key is logged once). Stage::any skips the stage check.
const GpuShader* gpu_shader(const void* bytes, std::size_t n, Stage stage);
const GpuShader* gpu_shader_by_key(uint64_t key);
// Record i in key order, i < gpu_db_count().
const GpuShader* gpu_db_at(std::size_t i);

}  // namespace shader_db
