#pragma once

// The geometry shader that completes the console's rectangle lists (rect.cpp).

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "x2w_shader/emit/container.h"

namespace x2w::vs {

// The geometry shader (a DXIL container) for rectangle lists drawn with the translated vertex
// shader `vs` and, if the pipeline has one, the translated pixel shader `ps` (both as
// HostShader::dxil). Its inputs and outputs are the vertex shader's outputs, register for
// register, so it links with both. A pixel shader that declares a primitive id gets the
// rectangle's number. Throws ShaderError.
std::vector<std::uint8_t> rect_list_gs(std::span<const std::uint8_t> vs,
                                       std::optional<std::span<const std::uint8_t>> ps);

// rect_list_gs from the vertex shader's outputs and the pixel shader's inputs.
std::vector<std::uint8_t> rect_build(std::vector<container::SigElement> outputs,
                                     std::span<const container::SigElement> ps_inputs);

}  // namespace x2w::vs
