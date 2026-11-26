// Shader database reader (WP1, vexport): the C1-C6 contract over X2W's shader database (src/shader_db),
// memory-mapped and thread-safe. gpu_db.cpp says where the database file is found.
#pragma once

#include "startup_stubs/gpu_abi.h"

namespace x2w_gpu {

// The C1 key of `bytes` (blake3, first 16 hex digits, lowercase), into out[17]. Implemented in gpu_db.cpp on the
// vendored reference code (startup_stubs/blake3/, portable only).
void gpu_key(const void* bytes, std::size_t size, char out[17]) noexcept;

// C5 lookup: the record for the bytecode as created at `stage`, or null (C6: the reader logs a miss once per key).
const GpuShader* gpu_shader(const void* bytes, std::size_t size, GpuStage stage) noexcept;
// The same by a key already computed (the creation slots compute it for the G0 key log).
const GpuShader* gpu_shader_by_key(const char key[17], GpuStage stage) noexcept;

// C3: the manifest of the loaded export, or null when none is loaded. The reader refuses an export whose
// view_heap_limit / raw_views disagree with abi:: and gpu_twins_on().
const GpuManifest* gpu_manifest() noexcept;

}  // namespace x2w_gpu
