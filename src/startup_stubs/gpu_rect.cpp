#include "startup_stubs/gpu_rect.h"
#include "startup_stubs/local_log.h"
#include "x2w_shader/emit/rect.h"
#include "x2w_shader/error.h"

namespace x2w_gpu {
std::vector<unsigned char> vp_rectangle_stage(D3D12_SHADER_BYTECODE vs, D3D12_SHADER_BYTECODE ps)
{
    const auto bytes = [](D3D12_SHADER_BYTECODE code) {
        return std::span(static_cast<const std::uint8_t*>(code.pShaderBytecode), code.BytecodeLength);
    };
    try {
        return x2w::vs::rect_list_gs(bytes(vs), ps.pShaderBytecode ? std::optional(bytes(ps)) : std::nullopt);
    } catch (const x2w::vs::ShaderError& e) {
        x2w_local::logf("x2w-gpu pipeline: no rectangle geometry stage: %s", e.what());
        return {};
    }
}
}
