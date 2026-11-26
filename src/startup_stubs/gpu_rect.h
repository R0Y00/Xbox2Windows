#pragma once

#include <d3d12.h>
#include <vector>

namespace x2w_gpu {
// Generate a geometry stage compatible with these translated stages. An unsupported
// signature leaves rectangle expansion to the draw path.
std::vector<unsigned char> vp_rectangle_stage(D3D12_SHADER_BYTECODE vs, D3D12_SHADER_BYTECODE ps);
}
