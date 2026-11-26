// Writes X2W's shader database (shader_db.h layout): the records a translator made, one per container.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "shader_db.h"

namespace shader_db {

// One record with its arrays owned (what the writer takes and gpu_shader() can be copied back into).
struct ShaderRecord {
    uint64_t key = 0, fnv64 = 0;
    Stage stage = Stage::vs;
    uint32_t container_bytes = 0;
    std::vector<uint8_t> dxil;
    uint32_t user_data_dwords = 0;
    std::vector<VxUserData> user_data;
    std::vector<std::array<uint32_t, 4>> samplers;
    std::vector<VxStaticSampler> static_samplers;
    std::vector<uint8_t> root_cbvs, root_srvs, root_uavs;
    std::vector<VxRootIndexedView> root_indexed_views;
    std::vector<VxUsed> used;
    bool has_threads = false;
    uint32_t threads[3] = {0, 0, 0};
    bool stores = false, stores_by_address = false, builder = false;
    uint32_t wave_size = 0, vgprs = 0;
};

ShaderRecord copy_record(const GpuShader& s);

// Writes `records` (sorted by key here; a repeated key is an error) and `manifest` to `path`, through a temporary
// file renamed over it, so a reader never maps a partial database.
bool write_db(const std::wstring& path, std::vector<ShaderRecord> records, const std::string& manifest, std::string& error);

// X2W's fnv_bytes (d3d12_draw.inl: the FNV-1a prime, basis 1469598103934665603), kept in each record beside the key.
uint64_t fnv64(const void* bytes, std::size_t n);

}  // namespace shader_db
