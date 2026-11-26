#pragma once

// Read signatures and vertex output interpolation without including emit.h and its front-end dependencies in rect.cpp or the
// tests.

#include <cstdint>
#include <span>
#include <vector>

#include "x2w_shader/emit/container.h"

namespace x2w::vs {

// The rows of an ISG1/OSG1 part. Throws ShaderError.
std::vector<container::SigElement> signature(std::span<const std::uint8_t> part);

// The title's vertex output interpolation lives in PSV0 (`data`: the part). A float packed beside
// an integer can be explicitly constant, as can a float on its own. OSG1 supplies the physical
// rows and columns used to match both tables. Throws ShaderError.
void vertex_output_interpolation(std::span<const std::uint8_t> data, std::span<container::SigElement> outputs);

}  // namespace x2w::vs
