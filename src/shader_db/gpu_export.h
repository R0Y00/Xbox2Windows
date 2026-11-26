// Reads an exported shader directory containing index.tsv, manifest.txt, and a JSON record and DXIL file for each
// key.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "shader_db_writer.h"

namespace shader_db {

struct ExportRow {
    ShaderRecord record;
    std::string first_source;  // first file containing the shader container
};

struct GpuExport {
    std::map<uint64_t, ExportRow> rows;  // by key
    std::string manifest;                // manifest.txt as written
    std::vector<std::string> errors;     // records that did not read
};

// Reads every record index.tsv names (`limit` records at most; 0 for all).
bool read_gpu_export(const std::wstring& dir, GpuExport& out, std::size_t limit = 0);

// One record from its JSON text and DXIL bytes; false with `why` if a field is missing or of the wrong kind.
bool record_from_json(const std::string& json, std::vector<uint8_t> dxil, ShaderRecord& out, std::string& why);

}  // namespace shader_db
