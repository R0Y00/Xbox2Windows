// When linked, x2w_shader supplies the per-container translator. Otherwise the tool reads translated records from
// the supplied export.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "shader_db_writer.h"

namespace shader_db {

struct Translator {
    const char* name;  // written to the manifest as `producer`
    // Fills every field of `out` but key, fnv64 and container_bytes (the caller's), or returns false with `why`.
    bool (*translate)(const uint8_t* container, std::size_t n, ShaderRecord& out, std::string& why);
    // Manifest lines (name value\n) for the settings the translations depend on.
    std::string (*settings)();
};

// The built-in translator, or null when the tool was built without x2w_shader.
const Translator* builtin_translator();

}  // namespace shader_db
